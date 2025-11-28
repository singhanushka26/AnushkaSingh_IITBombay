# ============================================================
#  BAJAJ DATATHON – RANK-1 BILL EXTRACTION PIPELINE (cv2-free)
#  EasyOCR + Tesseract version
#  Everything else = EXACT SAME as your provided code
# ============================================================

import os
import io
import re
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
from easyocr import Reader
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

# EasyOCR (Replacing PaddleOCR, keeping code identical otherwise)
easy_ocr = Reader(['en'], gpu=False, model_storage_directory='/app/models', download_enabled=True, detector='dbnet', recognizer='robustscanner', quantize=True)


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
#  IMAGE PREPROCESSING (cv2-free)
# ============================================================

def preprocess_image(img: Image.Image) -> Image.Image:
    """
    Same preprocessing as your version,
    just no cv2 and no paddlex required.
    """
    gray = img.convert("L")
    gray = ImageOps.autocontrast(gray)
    gray = gray.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    return gray.convert("RGB")


# ============================================================
#  HYBRID OCR (EasyOCR + Tesseract) - replaces PaddleOCR
# ============================================================

def run_hybrid_ocr(img: Image.Image) -> Tuple[str, List[Dict]]:
    """
    EXACT SAME structure as your previous function,
    but replacing PaddleOCR with EasyOCR.
    """

    # --- Tesseract ---
    try:
        tesseract_text = pytesseract.image_to_string(img)
    except Exception:
        tesseract_text = ""

    # --- EasyOCR ---
    try:
        easy_res = easy_ocr.readtext(np.array(img), detail=1)
    except Exception:
        easy_res = []

    boxes = []
    easy_texts = []

    for (bbox, text, conf) in easy_res:
        easy_texts.append(text)
        boxes.append({"text": text, "bbox": bbox})

    # Combine for LLM context (same behavior)
    combined = (tesseract_text or "") + "\n" + "\n".join(easy_texts)
    return combined.strip(), boxes


# ============================================================
#  FRAUD DETECTION (cv2-free)
# ============================================================

def fraud_score(img: Image.Image) -> float:
    gray = np.array(img.convert("L")).astype(float)
    gy, gx = np.gradient(gray)
    edge_mag = np.sqrt(gx**2 + gy**2)
    score = float(np.mean(edge_mag) / 50.0)
    return max(0.0, min(1.0, score))


# ============================================================
#  PAGE TYPE CLASSIFICATION
# ============================================================

def classify_page_type(ocr_text: str) -> str:
    t = ocr_text.lower()
    if any(x in t for x in ["pharma", "tablet", "inj", "tab", "caps"]):
        return "Pharmacy"
    if any(x in t for x in ["final bill", "summary", "net payable", "amount payable"]):
        return "Final Bill"
    return "Bill Detail"


# ============================================================
#  MULTI-CROP GENERATOR
# ============================================================

def generate_crops(img: Image.Image, n: int = 3) -> List[Image.Image]:
    w, h = img.size
    step = max(1, h // n)
    crops = []
    for i in range(n):
        top = i * step
        bottom = h if i == n - 1 else (i + 1) * step
        crops.append(img.crop((0, top, w, bottom)))
    return crops


# ============================================================
#  IMAGE → BASE64
# ============================================================

def image_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    data = buf.getvalue()
    q = 85
    while len(data) > 4 * 1024 * 1024 and q > 30:
        q -= 10
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=q)
        data = buf.getvalue()
    return "data:image/jpeg;base64," + base64.b64encode(data).decode()


# ============================================================
#  FASTAPI APP
# ============================================================

app = FastAPI(
    title="Bajaj Datathon – Hybrid Bill Extraction API (EasyOCR)",
    version="4.0.0",
    description=(
        "Hybrid Vision + EasyOCR + Tesseract + Fraud Detection + "
        "Multi-crop + Async Groq Inference + Semantic Dedupe (no cv2/libGL)"
    ),
)
# ============================================================
#  ADVANCED SYSTEM PROMPT (Same as your original)
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


# ============================================================
#  PAGE-TYPE SPECIFIC PROMPT BUILDER
# ============================================================

def build_system_prompt_for_page(page_type_hint: str) -> str:
    t = (page_type_hint or "Bill Detail").lower()
    extra = ""

    if "pharmacy" in t:
        extra = """
Pharmacy-specific hints:
- Items are mostly drug/medicine names, injections, syrups, tablets.
- Quantities may be strips, bottles, vials, ampoules, etc.
- Ignore batch numbers / expiry if noisy.
"""
    elif "final" in t:
        extra = """
Final Bill hints:
- Many totals present. Avoid extracting any total/tax rows.
- Page might have very few actual items.
"""
    else:
        extra = """
Bill Detail hints:
- Common categories: bed charges, investigations, pathology, procedures.
- Ignore section headers and subtitles.
"""

    return BASE_SYSTEM_PROMPT + "\n" + extra


# ============================================================
#  MODEL SELECTION (Same rules as your version)
# ============================================================

def choose_model(page_type: str, ocr_text: str, fraud: float) -> str:
    t = (page_type or "").lower()
    text = (ocr_text or "").lower()

    if "pharmacy" in t:
        return MODEL_SCOUT

    if any(k in text for k in ["rx", "caps", "syp", "inj", "tab"]) or fraud > 0.45:
        return MODEL_SCOUT

    return MODEL_MAVERICK


# ============================================================
#  ASYNC GROQ VISION CALL
# ============================================================

async def call_llm_vision(
    crop_b64: str,
    ocr_text: str,
    page_type_hint: str,
    model_name: str
):
    system_prompt = build_system_prompt_for_page(page_type_hint)

    # limit OCR context size
    snippet = (ocr_text or "").strip()
    if len(snippet) > 2000:
        snippet = snippet[:2000]

    user_text = f"""
You are extracting line items from ONE cropped region of a bill page.
Page type: {page_type_hint}

OCR TEXT (may contain noise):
{snippet}

Now silently analyze the crop + OCR and return STRICT JSON.
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
                            "detail": "auto"
                        },
                    ],
                },
            ],
        )

    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Groq API error: {e}"
        )

    # Parse token usage
    usage = getattr(response, "usage", None)
    if usage:
        token_usage = TokenUsage(
            total_tokens=int(usage.total_tokens or 0),
            input_tokens=int(usage.input_tokens or 0),
            output_tokens=int(usage.output_tokens or 0),
        )
    else:
        token_usage = TokenUsage(0, 0, 0)

    raw_text = response.output_text.strip()

    # robust JSON parsing
    try:
        parsed = json.loads(raw_text)
    except:
        f = raw_text.find("{")
        l = raw_text.rfind("}")
        parsed = json.loads(raw_text[f:l+1])

    return parsed, token_usage
# ============================================================
#  STRING UTILS – NUMERIC & FUZZY
# ============================================================

def _coerce_number(x: Any) -> float:
    if x is None:
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip().replace(",", "")
        if not s or s in {"-", "—", "NA", "N/A"}:
            return 0.0
        try:
            return float(s)
        except:
            return 0.0
    return 0.0


def _norm_name(name: str) -> str:
    s = (name or "").lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def fuzzy_ratio(a: str, b: str) -> float:
    import difflib
    return difflib.SequenceMatcher(None, a, b).ratio()


# ============================================================
#  HIGH-RECALL ROW RECONSTRUCTION USING OCR
# ============================================================

def reconstruct_missing_rows(
    llm_items: List[Dict[str, Any]],
    ocr_boxes: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    existing = {_norm_name(i.get("item_name", "")) for i in llm_items}
    new_items = []

    for box in ocr_boxes:
        text = (box.get("text") or "").strip()
        if len(text) < 3:
            continue

        lw = text.lower()
        if any(k in lw for k in ["total", "cgst", "sgst", "igst", "round", "gst", "tax", "sub total", "subtotal"]):
            continue

        norm = _norm_name(text)
        if not norm or norm in existing:
            continue

        existing.add(norm)
        new_items.append({
            "item_name": text,
            "item_amount": 0.0,
            "item_rate": 0.0,
            "item_quantity": 0.0
        })

    return llm_items + new_items


# ============================================================
#  OCR ALIGNMENT FOR QTY/RATE/AMOUNT
# ============================================================

def align_numbers_with_ocr(
    items: List[Dict[str, Any]],
    ocr_text: str
) -> List[Dict[str, Any]]:

    lines = [ln.strip() for ln in ocr_text.splitlines() if ln.strip()]
    out = []

    for it in items:
        name = it["item_name"]
        norm = _norm_name(name)

        qty = _coerce_number(it["item_quantity"])
        rate = _coerce_number(it["item_rate"])
        amt  = _coerce_number(it["item_amount"])

        # Already complete
        if qty > 0 and rate > 0 and amt > 0:
            out.append(it)
            continue

        # Find best matching OCR line
        best_line = None
        best_score = 0

        for ln in lines:
            score = fuzzy_ratio(norm, _norm_name(ln))
            if score > best_score:
                best_score = score
                best_line = ln

        if best_line and best_score > 0.6:
            nums = re.findall(r"\d+\.?\d*", best_line)
            nums_f = [float(n) for n in nums] if nums else []

            # heuristic mapping:
            # last → amt
            # previous → rate
            # first → qty
            if len(nums_f) >= 1 and amt == 0:
                amt = nums_f[-1]
            if len(nums_f) >= 2 and rate == 0:
                rate = nums_f[-2]
            if len(nums_f) >= 3 and qty == 0:
                qty = nums_f[0]

        out.append({
            "item_name": name,
            "item_amount": amt,
            "item_rate": rate,
            "item_quantity": qty
        })

    return out


# ============================================================
#  RECONCILE NUMERIC CONSISTENCY
# ============================================================

def reconcile_items_numeric(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    EPS = 0.01
    out = []

    for it in items:
        name = it["item_name"]
        qty  = _coerce_number(it["item_quantity"])
        rate = _coerce_number(it["item_rate"])
        amt  = _coerce_number(it["item_amount"])

        if rate and qty:
            computed = rate * qty
            if math.isfinite(computed) and abs(computed - amt) > EPS:
                amt = round(computed, 2)
        elif amt and qty and qty != 0:
            rate = round(amt / qty, 4)
        elif amt and not qty:
            qty = 1.0
            rate = amt

        out.append({
            "item_name": name.strip(),
            "item_amount": float(amt),
            "item_rate": float(rate),
            "item_quantity": float(qty)
        })

    return out


# ============================================================
#  SEMANTIC DEDUPLICATION (WITHIN PAGE)
# ============================================================

def semantic_dedupe_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped = []

    for it in items:
        name = _norm_name(it["item_name"])
        qty = it["item_quantity"]
        rate = it["item_rate"]
        amt = it["item_amount"]

        is_dup = False

        for ex in deduped:
            if (
                fuzzy_ratio(name, _norm_name(ex["item_name"])) > 0.88
                and abs(qty - ex["item_quantity"]) < 0.01
                and abs(rate - ex["item_rate"]) < 0.01
                and abs(amt - ex["item_amount"]) < 0.05
            ):
                is_dup = True
                break

        if not is_dup:
            deduped.append({
                "item_name": it["item_name"],
                "item_amount": amt,
                "item_rate": rate,
                "item_quantity": qty
            })

    return deduped
# ============================================================
#  MAIN ENDPOINT
# ============================================================

@app.post("/extract-bill-data", response_model=ExtractBillDataResponse)
async def extract_bill_data(req: ExtractBillDataRequest):
    """
    Datathon API:
    {
        "document": "<public PDF/image URL>"
    }
    """
    url = str(req.document)

    # 1. Download
    content = await download_file(url)
    mime = guess_mime_type(url, content)

    # 2. Load pages
    if mime == "application/pdf":
        try:
            pages = convert_from_bytes(content)
        except Exception as e:
            raise HTTPException(400, f"PDF conversion failed: {e}")
    else:
        try:
            pages = [Image.open(io.BytesIO(content)).convert("RGB")]
        except Exception as e:
            raise HTTPException(400, f"Image open failed: {e}")

    if not pages:
        return ExtractBillDataResponse(
            is_success=False,
            message="No pages found in document."
        )

    # Prepare for async LLM execution
    tasks = []
    meta = []
    global_usage = TokenUsage(0, 0, 0)

    page_infos = []  # store per-page OCR data

    # --------------------------------------------------------
    # 3. Process each page → OCR → crops → LLM tasks
    # --------------------------------------------------------
    for page_idx, raw_page in enumerate(pages, start=1):

        # Preprocess page
        img = preprocess_image(raw_page)

        # OCR + fraud
        ocr_text, ocr_boxes = run_hybrid_ocr(img)
        fscore = fraud_score(img)
        page_type = classify_page_type(ocr_text)
        model = choose_model(page_type, ocr_text, fscore)

        page_infos.append({
            "page_idx": page_idx,
            "img": img,
            "ocr_text": ocr_text,
            "ocr_boxes": ocr_boxes,
            "page_type": page_type,
            "model": model
        })

        # Multi-crop
        crops = generate_crops(
            img,
            n=3 if page_type != "Pharmacy" else 4
        )

        # enqueue LLM tasks
        for crop_idx, crop in enumerate(crops, start=1):
            crop_b64 = image_to_b64(crop)

            tasks.append(
                call_llm_vision(
                    crop_b64=crop_b64,
                    ocr_text=ocr_text,
                    page_type_hint=page_type,
                    model_name=model
                )
            )

            meta.append({
                "page_idx": page_idx,
                "crop_idx": crop_idx,
                "page_type": page_type,
                "ocr_text": ocr_text
            })

    if not tasks:
        return ExtractBillDataResponse(
            is_success=False,
            message="No crops generated."
        )

    # --------------------------------------------------------
    # 4. Run all LLM calls concurrently
    # --------------------------------------------------------
    results = await asyncio.gather(*tasks)

    # --------------------------------------------------------
    # 5. Aggregate by page
    # --------------------------------------------------------
    page_items_raw = {}
    text_by_page = {p["page_idx"]: p["ocr_text"] for p in page_infos}
    boxes_by_page = {p["page_idx"]: p["ocr_boxes"] for p in page_infos}
    type_by_page  = {p["page_idx"]: p["page_type"] for p in page_infos}

    for (parsed, usage), m in zip(results, meta):
        page_idx = m["page_idx"]

        # accumulate token usage
        global_usage.total_tokens += usage.total_tokens
        global_usage.input_tokens += usage.input_tokens
        global_usage.output_tokens += usage.output_tokens

        items = parsed.get("bill_items", []) or []

        if page_idx not in page_items_raw:
            page_items_raw[page_idx] = {
                "page_type": type_by_page[page_idx],
                "items": []
            }

        page_items_raw[page_idx]["items"].extend(items)

    # --------------------------------------------------------
    # 6. Page-level post-processing
    # --------------------------------------------------------
    final_pages = []

    for page_idx, info in page_items_raw.items():
        page_type = info["page_type"]
        items = info["items"]

        ocr_text  = text_by_page.get(page_idx, "")
        ocr_boxes = boxes_by_page.get(page_idx, [])

        # High recall
        items = reconstruct_missing_rows(items, ocr_boxes)

        # OCR alignment
        items = align_numbers_with_ocr(items, ocr_text)

        # Numeric reconcile
        items = reconcile_items_numeric(items)

        # Page-level dedupe
        items = semantic_dedupe_items(items)

        # Remove totals/tax lines again for safety
        cleaned = []
        for it in items:
            name = (it["item_name"] or "").strip().lower()
            if any(k in name for k in [
                "total", "cgst", "sgst", "igst", "round",
                "gst", "tax", "sub total", "subtotal"
            ]):
                continue

            cleaned.append(BillItem(
                item_name=it["item_name"].strip(),
                item_amount=_coerce_number(it["item_amount"]),
                item_rate=_coerce_number(it["item_rate"]),
                item_quantity=_coerce_number(it["item_quantity"])
            ))

        final_pages.append(PageItems(
            page_no=str(page_idx),
            page_type=page_type,
            bill_items=cleaned
        ))

    # --------------------------------------------------------
    # 7. Global dedupe across pages (avoid duplicate scan entries)
    # --------------------------------------------------------
    deduped_pages = []
    seen = []

    for pg in final_pages:
        new_items = []
        for it in pg.bill_items:
            dup = False
            for ex in seen:
                if (
                    fuzzy_ratio(_norm_name(it.item_name), _norm_name(ex.item_name)) > 0.9
                    and abs(it.item_rate - ex.item_rate) < 0.01
                    and abs(it.item_quantity - ex.item_quantity) < 0.01
                    and abs(it.item_amount - ex.item_amount) < 0.05
                ):
                    dup = True
                    break

            if not dup:
                seen.append(it)
                new_items.append(it)

        pg.bill_items = new_items
        deduped_pages.append(pg)

    # --------------------------------------------------------
    # 8. Final output
    # --------------------------------------------------------
    total_items = sum(len(p.bill_items) for p in deduped_pages)

    return ExtractBillDataResponse(
        is_success=True,
        token_usage=global_usage,
        data=ExtractBillDataResponseData(
            pagewise_line_items=deduped_pages,
            total_item_count=total_items
        )
    )
