# ============================================================
#  BAJAJ DATATHON – RANK-1 BILL EXTRACTION PIPELINE
#  Single-file version (Option A)
#  Includes:
#   - Multi-crop vision extraction
#   - PaddleOCR + Tesseract hybrid OCR
#   - Fraud detection
#   - Async Groq LLM inference
#   - Dynamic model switching (Maverick ↔ Scout)
#   - OCR alignment for qty/rate
#   - Semantic deduplication
#   - High-recall row reconstruction
# ============================================================

import os
import io
import re
import cv2
import json
import math
import base64
import httpx
import asyncio
import numpy as np
import mimetypes
from typing import List, Dict, Any, Tuple, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from PIL import Image, ImageFilter, ImageOps, ImageStat
from paddleocr import PaddleOCR
import pytesseract
from pdf2image import convert_from_bytes
from openai import AsyncOpenAI


# ============================================================
#  CONFIG
# ============================================================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Async Groq Client
groq_client = AsyncOpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# Two Vision Models
MODEL_MAVERICK = "meta-llama/llama-4-maverick-17b-128e-instruct"
MODEL_SCOUT = "meta-llama/llama-4-scout-17b-16e-instruct"

# PaddleOCR Instance
paddle_ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    show_log=False
)


# ============================================================
#  BASIC SCHEMAS (Datathon Spec)
# ============================================================

class BillItem(BaseModel):
    item_name: str
    item_amount: float
    item_rate: float
    item_quantity: float

class PageItems(BaseModel):
    page_no: str
    page_type: str
    bill_items: List[BillItem]

class ExtractBillDataRequest(BaseModel):
    document: HttpUrl

class ExtractBillDataResponseData(BaseModel):
    pagewise_line_items: List[PageItems]
    total_item_count: int

class TokenUsage(BaseModel):
    total_tokens: int
    input_tokens: int
    output_tokens: int

class ExtractBillDataResponse(BaseModel):
    is_success: bool
    token_usage: Optional[TokenUsage] = None
    data: Optional[ExtractBillDataResponseData] = None
    message: Optional[str] = None


# ============================================================
#  HELPER: DOWNLOAD FILE (ASYNC)
# ============================================================

async def download_file(url: str) -> bytes:
    try:
        async with httpx.AsyncClient(timeout=45) as client:
            r = await client.get(url)
            r.raise_for_status()
            return r.content
    except Exception as e:
        raise HTTPException(400, f"Document download failed: {e}")


# ============================================================
#  MIME GUESS
# ============================================================

def guess_mime_type(url: str, content: bytes) -> str:
    mime, _ = mimetypes.guess_type(url)
    if mime:
        return mime
    if content[:4] == b"%PDF":
        return "application/pdf"
    return "application/octet-stream"


# ============================================================
#  IMAGE PREPROCESSING
# ============================================================

def preprocess_image(img: Image.Image) -> Image.Image:
    """
    Improves OCR + Vision accuracy:
    - Deskew
    - Increase contrast
    - Sharpen edges
    - Remove noise
    """

    # Convert to OpenCV
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # --- Deskew ---
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    coords = np.column_stack(np.where(gray < 200))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        (h, w) = cv_img.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        cv_img = cv2.warpAffine(cv_img, M, (w, h), flags=cv2.INTER_CUBIC)

    # --- Sharpen ---
    sharpen = cv2.GaussianBlur(cv_img, (0,0), 3)
    cv_img = cv2.addWeighted(cv_img, 1.5, sharpen, -0.5, 0)

    # Convert back to PIL
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))


# ============================================================
#  HYBRID OCR (PaddleOCR + Tesseract)
# ============================================================

def run_hybrid_ocr(img: Image.Image) -> Tuple[str, List[Dict]]:
    """
    Returns:
        text (str)
        boxes (list of OCR bounding boxes)
    """

    # Tesseract for printed text
    try:
        tesseract_text = pytesseract.image_to_string(img)
    except:
        tesseract_text = ""

    # PaddleOCR for handwriting + complex layouts
    paddle_res = paddle_ocr.ocr(np.array(img), cls=True)
    paddle_texts = []
    boxes = []

    if paddle_res:
        for block in paddle_res:
            for line in block:
                txt = line[1][0]
                bbox = line[0]
                paddle_texts.append(txt)
                boxes.append({
                    "text": txt,
                    "bbox": bbox
                })

    combined = (tesseract_text or "") + "\n" + "\n".join(paddle_texts)
    return combined.strip(), boxes


# ============================================================
#  FRAUD DETECTION
# ============================================================

def fraud_score(img: Image.Image) -> float:
    """
    Detect overwritten totals, white patches, unnatural edits.
    """
    gray = np.array(img.convert("L")).astype(float)

    # High-frequency estimate using Laplacian
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    score = np.mean(np.abs(lap)) / 50.0
    return float(min(1.0, max(0.0, score)))


# ============================================================
#  PAGE TYPE CLASSIFICATION
# ============================================================

def classify_page_type(ocr_text: str) -> str:
    t = ocr_text.lower()

    if any(x in t for x in ["pharma", "dose", "tablet", "inj", "tab", "caps"]):
        return "Pharmacy"

    if any(x in t for x in ["final bill", "summary", "net payable", "amount payable"]):
        return "Final Bill"

    return "Bill Detail"


# ============================================================
#  MULTI-CROP GENERATOR
# ============================================================

def generate_crops(img: Image.Image, n: int = 3) -> List[Image.Image]:
    """
    Split image into N horizontal crops for higher recall.
    """
    w, h = img.size
    crops = []
    step = h // n
    for i in range(n):
        top = i * step
        bottom = h if i == n-1 else (i+1) * step
        crop = img.crop((0, top, w, bottom))
        crops.append(crop)
    return crops


# ============================================================
#  IMAGE → BASE64
# ============================================================

def image_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    data = buf.getvalue()
    q = 85
    while len(data) > 4*1024*1024 and q > 30:
        q -= 10
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=q)
        data = buf.getvalue()
    return "data:image/jpeg;base64," + base64.b64encode(data).decode()

# ============================================================
#  FASTAPI APP
# ============================================================

app = FastAPI(
    title="Bajaj Datathon – Hybrid Bill Extraction API",
    version="4.0.0",
    description=(
        "Hybrid Vision + PaddleOCR + Tesseract + Fraud Detection + "
        "Multi-crop + Async Groq Inference + Semantic Dedupe"
    ),
)


# ============================================================
#  BLANK PAGE DETECTION
# ============================================================

def is_blank_page(img: Image.Image, ocr_text: str) -> bool:
    gray = img.convert("L")
    stat = ImageStat.Stat(gray)
    mean = stat.mean[0] if stat.mean else 255.0
    if mean > 245 and len(ocr_text.strip()) < 30:
        return True
    return False


# ============================================================
#  STRING UTILS – NUMERIC & FUZZY
# ============================================================

def _coerce_number(x: Any) -> float:
    if x is None:
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if s == "" or s in {"-", "—", "NA", "N/A"}:
            return 0.0
        s = s.replace(",", "")
        try:
            return float(s)
        except Exception:
            return 0.0
    return 0.0


def _norm_name(name: str) -> str:
    s = (name or "").lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def fuzzy_ratio(a: str, b: str) -> float:
    """
    Cheap semantic similarity using difflib.
    """
    import difflib
    return difflib.SequenceMatcher(None, a, b).ratio()


# ============================================================
#  ADVANCED SYSTEM PROMPT (3 PAGE TYPES)
# ============================================================

BASE_SYSTEM_PROMPT = """
You are an advanced medical bill extraction engine.

Goal:
From a SINGLE PAGE IMAGE (or cropped region of a page) of a bill/invoice,
extract ONLY item-level rows from the charge/medicine tables and output
STRICT JSON according to the schema. This cropped region belongs to exactly
one page of a multi-page bill.

Definitions:
- A "line item" is ONE row (or logical row) describing a test, procedure,
  bed/room charge, consultation, or medicine. It usually has:
  * Description (name of test/procedure/medicine/bed)
  * Quantity
  * Rate (per-unit price)
  * Total/Net amount for that row.

- Ignore patient details, doctor names, hospital headers, logos,
  addresses, and global bill summary text.
- Ignore section titles like:
  "Radiological Investigation", "CONSULTATION", "BED CHARGES",
  "PHARMACY CHARGE", "PATHOLOGY", etc.
  These are NOT items.

Very important:
- Every JSON row must correspond to a real, visible row in THIS cropped region.
- Do NOT invent rows from other pages or crops.
- If an item row is cut across two crops, you may still extract it from either
  crop, but do NOT duplicate.

Page types:
- "Bill Detail": tests / procedures / bed / investigations line items.
- "Final Bill": summary-style page, may contain a few line items plus grand total.
- "Pharmacy": medicines, tablets, injections, bottles, strips.

You MAY be given OCR text for the same region. This OCR can be noisy,
but it is useful to read very small or blurred numbers. When OCR text is
provided, prefer exact numbers from OCR if they clearly match the row.

Numeric rules:
- item_quantity = numeric value under columns like "Qty", "QTY/Hrs", "QTY".
- item_rate     = "Rate", "RATE", "Price", "Per-unit price".
- item_amount   = "Amount", "Gross", "Net", "Total" for that row.
- If quantity is missing but a single total is clearly shown for that row,
  assume quantity = 1 and rate = total.
- If rate is missing but quantity and amount are visible, set rate = amount / quantity.
- If a numeric field is not visible / unreadable, set it to 0.0 instead of null.

DO NOT include summary rows / totals as items:
- Lines containing words like: "TOTAL", "SUB TOTAL", "SUBTOTAL",
  "NET AMOUNT", "AMOUNT PAYABLE", "ROUND OFF", "CGST", "SGST",
  "IGST", "TAX", "DISC" (discount) must NOT be extracted as line items.

Self-check before final output:
1. Remove any row whose description clearly indicates tax, discount, or totals.
2. Ensure item_amount ≈ item_rate * item_quantity if all are non-zero.
   If there is a small mismatch, prefer arithmetic: recompute item_amount
   as rate * quantity for that row.
3. Never merge two distinct items into one JSON object.

Output format:
Output ONLY JSON, no markdown, no comments, with shape:

{
  "page_type": "<Bill Detail | Final Bill | Pharmacy>",
  "bill_items": [
    {
      "item_name": "<string>",
      "item_amount": <float>,
      "item_rate": <float>,
      "item_quantity": <float>
    }
  ]
}

Restrictions:
- Do NOT include any extra top-level fields.
- Do NOT wrap JSON in ``` fences.
- Do NOT output intermediate reasoning; just apply it silently.
"""


def build_system_prompt_for_page(page_type_hint: str) -> str:
    """
    Slightly adjusts base prompt depending on page type.
    """
    t = (page_type_hint or "Bill Detail").strip()
    t_low = t.lower()
    extra = ""

    if "pharmacy" in t_low:
        extra = """
Pharmacy-specific hints:
- Items are mostly drug/medicine names, injections, syrups, tablets.
- Quantities can be strips, bottles, vials, ampoules, etc.
- Row often has: Drug name, Batch no, Expiry, Qty, Rate, Amount.
- Ignore batch numbers and expiry in item_name if they clutter the name.
"""
    elif "final" in t_low:
        extra = """
Final Bill hints:
- Page may have very few line items and many totals.
- Be extra careful not to treat grand totals or section totals as items.
"""
    else:
        extra = """
Bill Detail hints:
- Focus on tests, procedures, bed charges, investigations.
- Section subtitles are not items.
"""

    return BASE_SYSTEM_PROMPT + "\n" + extra


# ============================================================
#  MODEL SELECTION
# ============================================================

def choose_model(page_type: str, ocr_text: str, fraud: float) -> str:
    """
    Heuristic:
    - Use SCOUT for noisy/handwritten/pharmacy-style pages.
    - Use MAVERICK for clean tabular pages.
    """
    t = (page_type or "").lower()
    text = (ocr_text or "").lower()

    # Pharmacy / messy pages → Scout
    if "pharmacy" in t:
        return MODEL_SCOUT

    # Heavy handwriting / noisy
    if any(k in text for k in ["rx", "caps", "syp", "inj", "tab"]) or fraud > 0.45:
        return MODEL_SCOUT

    # Default
    return MODEL_MAVERICK


# ============================================================
#  VISION + LLM CALL (ASYNC)
# ============================================================

async def call_llm_vision(
    crop_b64: str,
    ocr_text: str,
    page_type_hint: str,
    model_name: str
) -> Tuple[Dict[str, Any], TokenUsage]:
    """
    Calls Groq vision model with advanced prompt.
    """
    system_prompt = build_system_prompt_for_page(page_type_hint)
    # Truncate OCR text to reasonable length
    ocr_snippet = (ocr_text or "").strip()
    if len(ocr_snippet) > 2000:
        ocr_snippet = ocr_snippet[:2000]

    user_text = f"""
You are extracting line items from ONE cropped region of a single bill page.

This crop belongs to a page of type: {page_type_hint}.

Below is OCR text from this exact region (may contain noise):

OCR_TEXT_START
{ocr_snippet}
OCR_TEXT_END

Now silently process the crop image and OCR together, and output the JSON.
    """.strip()

    try:
        response = await groq_client.responses.create(
            model=model_name,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_text},
                        {
                            "type": "input_image",
                            "image_url": crop_b64,
                            "detail": "auto",
                        },
                    ],
                },
            ],
        )
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Groq API error: {e}",
        )

    usage = getattr(response, "usage", None)
    if usage is not None:
        token_usage = TokenUsage(
            total_tokens=int(usage.total_tokens or 0),
            input_tokens=int(usage.input_tokens or 0),
            output_tokens=int(usage.output_tokens or 0),
        )
    else:
        token_usage = TokenUsage(total_tokens=0, input_tokens=0, output_tokens=0)

    raw_text = response.output_text.strip()

    # Robust JSON parsing
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        first = raw_text.find("{")
        last = raw_text.rfind("}")
        if first != -1 and last != -1 and last > first:
            json_str = raw_text[first:last + 1]
            parsed = json.loads(json_str)
        else:
            raise HTTPException(
                status_code=500,
                detail=f"LLM response not valid JSON: {raw_text[:200]}",
            )

    return parsed, token_usage


# ============================================================
#  HIGH-RECALL ROW RECONSTRUCTION USING OCR
# ============================================================

def reconstruct_missing_rows(
    llm_items: List[Dict[str, Any]],
    ocr_boxes: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Use OCR bounding boxes to add candidate items that vision LLM missed.
    We only add items whose text looks like a plausible item_name and
    doesn't already exist in llm_items.
    """
    existing_names = {_norm_name(i.get("item_name", "")) for i in llm_items}
    new_items: List[Dict[str, Any]] = []

    for box in ocr_boxes:
        text = (box.get("text") or "").strip()
        if len(text) < 3:
            continue

        lower = text.lower()
        # Skip total/tax-like lines
        if any(kw in lower for kw in ["total", "cgst", "sgst", "igst", "round", "gst", "tax", "sub total", "subtotal"]):
            continue

        norm = _norm_name(text)
        if not norm or norm in existing_names:
            continue

        # Looks like a reasonable item candidate
        existing_names.add(norm)
        new_items.append(
            {
                "item_name": text,
                "item_amount": 0.0,
                "item_rate": 0.0,
                "item_quantity": 0.0,
            }
        )

    return llm_items + new_items


# ============================================================
#  OCR ALIGNMENT FOR QTY/RATE/AMOUNT
# ============================================================

def align_numbers_with_ocr(
    items: List[Dict[str, Any]],
    ocr_text: str
) -> List[Dict[str, Any]]:
    """
    For items with missing rate/qty/amount, try to fill from OCR text.
    Heuristic:
    - Find line in OCR that best matches item_name.
    - Extract trailing numbers (qty, rate, amount).
    """

    lines = [ln.strip() for ln in (ocr_text or "").splitlines() if ln.strip()]
    result: List[Dict[str, Any]] = []

    for it in items:
        name = it.get("item_name", "")
        norm_name = _norm_name(name)
        qty = _coerce_number(it.get("item_quantity"))
        rate = _coerce_number(it.get("item_rate"))
        amt = _coerce_number(it.get("item_amount"))

        # Only bother when some numbers are missing/zero
        if qty > 0 and rate > 0 and amt > 0:
            result.append(
                {
                    "item_name": name,
                    "item_amount": amt,
                    "item_rate": rate,
                    "item_quantity": qty,
                }
            )
            continue

        best_line = None
        best_score = 0.0

        for ln in lines:
            score = fuzzy_ratio(norm_name, _norm_name(ln))
            if score > best_score:
                best_score = score
                best_line = ln

        if best_line and best_score > 0.6:
            # Extract numbers from this line
            nums = re.findall(r"\d+\.?\d*", best_line)
            nums_f = [float(x) for x in nums] if nums else []

            # Very rough heuristic: last = amount, previous = rate, first = qty
            if len(nums_f) >= 1 and amt == 0.0:
                amt = nums_f[-1]
            if len(nums_f) >= 2 and rate == 0.0:
                rate = nums_f[-2]
            if len(nums_f) >= 3 and qty == 0.0:
                qty = nums_f[0]

        result.append(
            {
                "item_name": name,
                "item_amount": amt,
                "item_rate": rate,
                "item_quantity": qty,
            }
        )

    return result


# ============================================================
#  RECONCILE NUMERIC CONSISTENCY
# ============================================================

def reconcile_items_numeric(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    EPS = 0.01
    out: List[Dict[str, Any]] = []

    for it in items:
        name = it.get("item_name", "")
        qty = _coerce_number(it.get("item_quantity"))
        rate = _coerce_number(it.get("item_rate"))
        amt = _coerce_number(it.get("item_amount"))

        if rate and qty:
            computed = rate * qty
            if math.isfinite(computed) and abs(computed - amt) > EPS:
                amt = round(computed, 2)
        elif amt and qty and qty != 0:
            rate = round(amt / qty, 4)
        elif amt and (not qty or qty == 0):
            qty = 1.0
            rate = round(amt, 2)

        out.append(
            {
                "item_name": name.strip(),
                "item_amount": float(amt),
                "item_rate": float(rate),
                "item_quantity": float(qty),
            }
        )

    return out


# ============================================================
#  SEMANTIC DEDUPLICATION (ACROSS ITEMS)
# ============================================================

def semantic_dedupe_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicates using semantic name similarity + numeric closeness.
    """
    deduped: List[Dict[str, Any]] = []
    for it in items:
        name = _norm_name(it.get("item_name", ""))
        qty = _coerce_number(it.get("item_quantity"))
        rate = _coerce_number(it.get("item_rate"))
        amt = _coerce_number(it.get("item_amount"))

        is_dup = False
        for existing in deduped:
            ename = _norm_name(existing["item_name"])
            eqty = existing["item_quantity"]
            erate = existing["item_rate"]
            eamt = existing["item_amount"]

            name_sim = fuzzy_ratio(name, ename)
            if name_sim > 0.88 and \
               abs(qty - eqty) < 0.01 and \
               abs(rate - erate) < 0.01 and \
               abs(amt - eamt) < 0.05:
                is_dup = True
                break

        if not is_dup:
            deduped.append(
                {
                    "item_name": it["item_name"],
                    "item_amount": amt,
                    "item_rate": rate,
                    "item_quantity": qty,
                }
            )

    return deduped


# ============================================================
#  MAIN ENDPOINT
# ============================================================

@app.post("/extract-bill-data", response_model=ExtractBillDataResponse)
async def extract_bill_data(req: ExtractBillDataRequest):
    """
    Datathon API:
      Input:  { "document": "<public PDF/image URL>" }
      Output: is_success, token_usage, data.pagewise_line_items, data.total_item_count
    """

    url_str = str(req.document)

    # 1. Download
    content = await download_file(url_str)
    mime = guess_mime_type(url_str, content)

    # 2. Load pages
    pil_pages: List[Image.Image] = []
    if mime == "application/pdf":
        try:
            pil_pages = convert_from_bytes(content)
        except Exception as e:
            raise HTTPException(400, f"PDF to image conversion failed: {e}")
    else:
        try:
            img = Image.open(io.BytesIO(content)).convert("RGB")
        except Exception as e:
            raise HTTPException(400, f"Unable to open image: {e}")
        pil_pages = [img]

    if not pil_pages:
        return ExtractBillDataResponse(
            is_success=False,
            message="No pages could be extracted from the document."
        )

    # 3. Prepare tasks for async LLM calls
    tasks = []
    meta: List[Dict[str, Any]] = []  # store page_idx, crop_idx, page_type, ocr_text
    global_token_usage = TokenUsage(total_tokens=0, input_tokens=0, output_tokens=0)

    # Keep OCR & fraud per PAGE (not per crop)
    page_infos: List[Dict[str, Any]] = []

    for page_idx, raw_img in enumerate(pil_pages, start=1):
        # Preprocess page
        img = preprocess_image(raw_img)

        # OCR + fraud
        ocr_text, ocr_boxes = run_hybrid_ocr(img)
        if is_blank_page(img, ocr_text):
            continue

        fscore = fraud_score(img)
        page_type = classify_page_type(ocr_text)
        model_name = choose_model(page_type, ocr_text, fscore)

        page_infos.append(
            {
                "page_idx": page_idx,
                "img": img,
                "ocr_text": ocr_text,
                "ocr_boxes": ocr_boxes,
                "fraud_score": fscore,
                "page_type": page_type,
                "model_name": model_name,
            }
        )

        # Multi-crop for this page
        crops = generate_crops(img, n=3 if page_type != "Pharmacy" else 4)
        for crop_idx, crop_img in enumerate(crops, start=1):
            crop_b64 = image_to_b64(crop_img)

            # For simplicity, we reuse full-page OCR text; could be refined to crop-only.
            task = call_llm_vision(
                crop_b64=crop_b64,
                ocr_text=ocr_text,
                page_type_hint=page_type,
                model_name=model_name,
            )
            tasks.append(task)
            meta.append(
                {
                    "page_idx": page_idx,
                    "crop_idx": crop_idx,
                    "page_type": page_type,
                    "ocr_text": ocr_text,
                }
            )

    if not tasks:
        return ExtractBillDataResponse(
            is_success=False,
            message="All pages detected as blank; nothing to extract."
        )

    # 4. Run all LLM calls concurrently
    llm_results = await asyncio.gather(*tasks)

    # 5. Aggregate raw items per page
    page_items_raw: Dict[int, Dict[str, Any]] = {}  # page_idx -> { "page_type": str, "items": [...], "ocr_text": str, "ocr_boxes": [...] }

    # Map page_idx → ocr_boxes
    boxes_by_page = {info["page_idx"]: info["ocr_boxes"] for info in page_infos}
    text_by_page = {info["page_idx"]: info["ocr_text"] for info in page_infos}
    type_by_page = {info["page_idx"]: info["page_type"] for info in page_infos}

    for (parsed, usage), m in zip(llm_results, meta):
        page_idx = m["page_idx"]
        page_type = m["page_type"]

        global_token_usage.total_tokens += usage.total_tokens
        global_token_usage.input_tokens += usage.input_tokens
        global_token_usage.output_tokens += usage.output_tokens

        bill_items = parsed.get("bill_items", []) or []

        if page_idx not in page_items_raw:
            page_items_raw[page_idx] = {
                "page_type": page_type,
                "items": [],
            }

        page_items_raw[page_idx]["items"].extend(bill_items)

    # 6. Build final PageItems list
    final_pages: List[PageItems] = []
    for page_idx, info in page_items_raw.items():
        page_type = info["page_type"]
        raw_items = info["items"]
        ocr_text = text_by_page.get(page_idx, "")
        ocr_boxes = boxes_by_page.get(page_idx, [])

        # 6a. High-recall reconstruction
        raw_items = reconstruct_missing_rows(raw_items, ocr_boxes)

        # 6b. OCR alignment
        raw_items = align_numbers_with_ocr(raw_items, ocr_text)

        # 6c. Numeric reconcile
        raw_items = reconcile_items_numeric(raw_items)

        # 6d. Semantic dedupe within page
        raw_items = semantic_dedupe_items(raw_items)

        # 6e. Filter out rows that are clearly totals/taxes again (safety)
        cleaned_items: List[BillItem] = []
        for it in raw_items:
            name = (it.get("item_name") or "").strip()
            lower = name.lower()
            if any(kw in lower for kw in ["total", "cgst", "sgst", "igst", "round", "gst", "tax", "sub total", "subtotal"]):
                continue

            cleaned_items.append(
                BillItem(
                    item_name=name,
                    item_amount=float(_coerce_number(it.get("item_amount"))),
                    item_rate=float(_coerce_number(it.get("item_rate"))),
                    item_quantity=float(_coerce_number(it.get("item_quantity"))),
                )
            )

        page_obj = PageItems(
            page_no=str(page_idx),
            page_type=page_type,
            bill_items=cleaned_items,
        )
        final_pages.append(page_obj)

    # 7. Global semantic dedupe across pages (to avoid duplicates from multi-scan issues)
    deduped_pages: List[PageItems] = []
    seen_global: List[BillItem] = []

    for p in final_pages:
        new_items: List[BillItem] = []
        for it in p.bill_items:
            is_dup = False
            for ex in seen_global:
                name_sim = fuzzy_ratio(_norm_name(it.item_name), _norm_name(ex.item_name))
                if name_sim > 0.9 and \
                   abs(it.item_rate - ex.item_rate) < 0.01 and \
                   abs(it.item_quantity - ex.item_quantity) < 0.01 and \
                   abs(it.item_amount - ex.item_amount) < 0.05:
                    is_dup = True
                    break
            if not is_dup:
                seen_global.append(it)
                new_items.append(it)

        p.bill_items = new_items
        deduped_pages.append(p)

    # 8. Aggregate
    total_items = sum(len(p.bill_items) for p in deduped_pages)
    data = ExtractBillDataResponseData(
        pagewise_line_items=deduped_pages,
        total_item_count=total_items,
    )

    return ExtractBillDataResponse(
        is_success=True,
        token_usage=global_token_usage,
        data=data,
    )
