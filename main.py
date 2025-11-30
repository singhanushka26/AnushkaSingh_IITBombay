# main.py
import base64
import io
import json
import math
import mimetypes
import os
import time
from collections import defaultdict
from typing import List, Optional, Tuple, Any, Dict

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from pdf2image import convert_from_bytes
from PIL import Image, ImageEnhance
from openai import OpenAI

# Optional OCR – will be used if available
try:
    import pytesseract

    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# ============================================================
#  Groq Client (OpenAI-compatible)
# ============================================================

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

# Fast bulk model
GROQ_VISION_MODEL_SCOUT = os.environ.get(
    "GROQ_VISION_MODEL_SCOUT",
    "meta-llama/llama-4-scout-17b-16e-instruct",
)

# Accurate refinement / heavy model
GROQ_VISION_MODEL_MAVERICK = os.environ.get(
    "GROQ_VISION_MODEL_MAVERICK",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
)

# Groq Vision limit: MAX 5 images per request
MAX_IMAGES_PER_REQUEST = 5

# Hard upper time budget (sec) – used only for strategy choice heuristics
GLOBAL_TIME_BUDGET_SEC = 120.0

# ============================================================
#  Pydantic Schemas – EXACT Datathon spec
# ============================================================

class BillItem(BaseModel):
    item_name: str
    item_amount: float
    item_rate: float
    item_quantity: float


class PageItems(BaseModel):
    page_no: str
    page_type: str  # "Bill Detail" | "Final Bill" | "Pharmacy"
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
#  FastAPI App
# ============================================================

app = FastAPI(
    title="Bajaj Datathon Bill Extraction API",
    version="9.0.0",
    description=(
        "Balanced bill extraction: dynamic strategy using Scout+Maverick, "
        "fast preprocessing and selective refinement to maximize accuracy "
        "under a ~120s global budget."
    ),
)

# ============================================================
#  Helpers – Download & Document Loading
# ============================================================

def download_document(url: str) -> bytes:
    """Download the document from the given URL."""
    try:
        resp = requests.get(url, timeout=40)
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to download document: {e}",
        )


def guess_mime_type(url: str, content: bytes) -> str:
    """Guess mime type based on URL extension or magic bytes."""
    mime, _ = mimetypes.guess_type(url)
    if mime:
        return mime
    if content[:4] == b"%PDF":
        return "application/pdf"
    return "application/octet-stream"


def load_document_pages(url: str, content: bytes) -> List[Image.Image]:
    """
    Load a document (PDF or image) into a list of RGB PIL Images.
    No resizing/cropping/enhancement done here – that is strategy dependent.
    """
    mime = guess_mime_type(url, content)
    pages: List[Image.Image] = []

    # Single images
    if mime.startswith("image/"):
        try:
            img = Image.open(io.BytesIO(content)).convert("RGB")
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Unable to open image document: {e}",
            )
        pages.append(img)
        return pages

    # PDFs
    if mime == "application/pdf":
        try:
            pdf_pages = convert_from_bytes(content)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Unable to convert PDF to images. "
                    "Ensure poppler is installed and available in PATH. "
                    f"Error: {e}"
                ),
            )
        for p in pdf_pages:
            pages.append(p.convert("RGB"))
        return pages

    # Fallback: try as image anyway
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported document type: {mime}",
        )
    pages.append(img)
    return pages


# ============================================================
#  Image Preprocessing (crop + enhance + resize + data_url)
# ============================================================

def smart_crop(img: Image.Image) -> Image.Image:
    """
    Simple heuristic crop to remove headers/footers and side margins.

    Balanced: keep more content to avoid missing top rows / totals.
    - Crop ~10% top, ~6% bottom, ~4% left/right.
    """
    w, h = img.size
    top = int(0.10 * h)
    bottom = int(0.94 * h)
    left = int(0.04 * w)
    right = int(0.96 * w)
    if bottom <= top or right <= left:
        return img
    return img.crop((left, top, right, bottom))


def enhance_image(
    img: Image.Image,
    contrast_factor: float,
    sharpness_factor: float,
) -> Image.Image:
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)
    img = ImageEnhance.Sharpness(img).enhance(sharpness_factor)
    return img


def resize_image_max_dim(img: Image.Image, max_dim: int) -> Image.Image:
    w, h = img.size
    scale = max(w, h) / float(max_dim)
    if scale <= 1.0:
        return img
    new_w = int(w / scale)
    new_h = int(h / scale)
    return img.resize((new_w, new_h), Image.LANCZOS)


def image_to_jpeg_bytes(
    img: Image.Image,
    quality: int = 60,
    max_bytes: int = 4 * 1024 * 1024,
) -> bytes:
    """Encode image to JPEG with given quality, shrinking if too large."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b = buf.getvalue()
    while len(b) > max_bytes and quality > 30:
        quality -= 10
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        b = buf.getvalue()
    return b


def jpeg_bytes_to_data_url(b: bytes) -> str:
    b64 = base64.b64encode(b).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# ============================================================
#  OCR – Rough row estimation
# ============================================================

def estimate_table_rows(img: Image.Image) -> int:
    """
    Use OCR to roughly estimate how many 'rows' of content exist.
    We only need this as a *signal* to detect under-extracted pages.

    Returns:
        int: approximate row count (lines that have some text+digits).
    """
    if not OCR_AVAILABLE:
        return 0

    try:
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    except Exception:
        return 0

    n = len(data.get("text", []))
    if n == 0:
        return 0

    rows = {}
    for i in range(n):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        line_num = data.get("line_num", [0] * n)[i]
        if line_num not in rows:
            rows[line_num] = {"has_digit": False, "has_alpha": False}
        if any(ch.isdigit() for ch in text):
            rows[line_num]["has_digit"] = True
        if any(ch.isalpha() for ch in text):
            rows[line_num]["has_alpha"] = True

    count = 0
    for info in rows.values():
        if info["has_digit"] and info["has_alpha"]:
            count += 1
    return count


# ============================================================
#  Dynamic Strategy Selection (Time-aware, Balanced)
# ============================================================

def choose_strategy(num_pages: int) -> Dict[str, Any]:
    """
    Choose strategy based on number of pages and a global 120s budget.

    Balanced design:
    - <= 6 pages: heavy Maverick, high res (max accuracy).
    - 7–10 pages: Maverick bulk, small batch, moderate res.
    - 11–15 pages: Maverick bulk, larger batch, slightly lower res.
    - > 15 pages: Scout bulk, small refinement window, lower res.

    OCR is used only when it won't blow up CPU time.
    """
    per_page_budget = GLOBAL_TIME_BUDGET_SEC / max(1, num_pages)

    # Base defaults (fast-ish)
    strategy: Dict[str, Any] = {
        "bulk_model": GROQ_VISION_MODEL_SCOUT,
        "bulk_batch_size": 3,
        "bulk_max_dim": 900,
        "contrast": 1.5,
        "sharpness": 1.35,
        "jpeg_quality": 60,
        "use_smart_crop": True,
        "use_ocr": False,      # enable below conditionally
        "use_refine": True,
        "refine_model": GROQ_VISION_MODEL_MAVERICK,
        "refine_limit": 6,
    }

    # Very small docs – go heavy
    if num_pages <= 6:
        strategy.update(
            {
                "bulk_model": GROQ_VISION_MODEL_MAVERICK,
                "bulk_batch_size": 1,
                "bulk_max_dim": 1050,
                "contrast": 1.6,
                "sharpness": 1.45,
                "use_ocr": True,
                "use_refine": False,  # already using Maverick on every page
                "refine_limit": 0,
            }
        )
    # Small/medium docs (sweet spot for 30–45s)
    elif num_pages <= 10:
        strategy.update(
            {
                "bulk_model": GROQ_VISION_MODEL_MAVERICK,
                "bulk_batch_size": 2,
                "bulk_max_dim": 950,
                "contrast": 1.55,
                "sharpness": 1.4,
                "use_ocr": True,   # helps find under-extracted pages
                "use_refine": True,
                "refine_limit": min(4, num_pages),  # refine 3–4 pages max
            }
        )
    # Medium docs
    elif num_pages <= 15:
        strategy.update(
            {
                "bulk_model": GROQ_VISION_MODEL_MAVERICK,
                "bulk_batch_size": 3,
                "bulk_max_dim": 900,
                "contrast": 1.5,
                "sharpness": 1.35,
                "use_ocr": num_pages <= 12,  # OCR only for <=12 pages
                "use_refine": True,
                "refine_limit": min(5, num_pages // 2),
            }
        )
    # Large docs – speed first, refine only a few pages
    else:
        strategy.update(
            {
                "bulk_model": GROQ_VISION_MODEL_SCOUT,
                "bulk_batch_size": 4 if per_page_budget < 5 else 3,
                "bulk_max_dim": 850,
                "contrast": 1.45,
                "sharpness": 1.3,
                "use_ocr": False,      # OCR is CPU heavy for big PDFs
                "use_refine": True,
                "refine_limit": 4,     # refine a few worst pages only
            }
        )

    # Always respect Groq's batch limit
    strategy["bulk_batch_size"] = min(strategy["bulk_batch_size"], MAX_IMAGES_PER_REQUEST)
    return strategy


def build_page_infos(
    raw_pages: List[Image.Image],
    strategy: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Apply cropping, enhancement, resizing, OCR and JPEG encoding
    to all pages according to the chosen strategy.

    Returns:
        List[Dict]: each entry has:
            - page_index: int
            - data_url: str
            - ocr_rows: int (approx row count)
    """
    page_infos: List[Dict[str, Any]] = []
    use_ocr = strategy.get("use_ocr", False)

    for idx, img in enumerate(raw_pages):
        img_proc = img.convert("RGB")

        if strategy.get("use_smart_crop", True):
            img_proc = smart_crop(img_proc)

        img_proc = enhance_image(
            img_proc,
            contrast_factor=strategy.get("contrast", 1.5),
            sharpness_factor=strategy.get("sharpness", 1.35),
        )
        img_proc = resize_image_max_dim(
            img_proc, strategy.get("bulk_max_dim", 900)
        )

        # OCR row estimation (optional)
        ocr_rows = 0
        if use_ocr and OCR_AVAILABLE:
            ocr_rows = estimate_table_rows(img_proc)

        jpeg_bytes = image_to_jpeg_bytes(
            img_proc,
            quality=strategy.get("jpeg_quality", 60),
        )
        data_url = jpeg_bytes_to_data_url(jpeg_bytes)

        page_infos.append(
            {
                "page_index": idx,
                "data_url": data_url,
                "ocr_rows": ocr_rows,
            }
        )
    return page_infos


# ============================================================
#  LLM Prompts – Bulk + Refinement
# ============================================================

SYSTEM_PROMPT_BULK = """
You are an expert medical BILL ITEM extraction engine for hospital bills.

Your goals:

1) Capture EVERY genuine line item from the tables (high recall).
2) Do NOT double-count or duplicate the same line item when it is only a summary.
3) Make the sum of all `item_amount` values across all pages as close as possible
   to the FINAL TOTAL printed in the bill.
4) Respect the exact JSON structure required by the HackRx Datathon.

CRITICAL BEHAVIOUR FOR REPEATED ROWS:

- If the same row appears MANY TIMES (e.g. many rows with
  "IP CONSULTATION CHARGES  Qty 1  Rate 1000  Amount 1000")
  then you MUST output ONE BILL ITEM PER VISUAL ROW.
  Example: if 20 such rows are visible, you must output 20 bill_items
  with the same name & numbers (do NOT collapse them into one).

- Section totals such as "TOTAL", "SUB TOTAL", "GRAND TOTAL",
  "NET AMOUNT PAYABLE", etc. MUST NOT be emitted as bill_items.

NUMERIC RULES:

- item_quantity: read from columns like "Qty", "No. of Days", etc.
- item_rate:     from "Rate", "Charges per day", etc.
- item_amount:   from "Amount", "Net Amt", etc.

If some numeric fields are missing for a row BUT other rows with the
same description show clear numbers, use that pattern:

  • If at least one row for the same item_name has
        quantity = q0 and amount = a0 (or rate = r0),
    then for rows where the numeric values are unreadable you may assume:
        quantity = q0
        rate     = a0 / q0 (or r0)
        amount   = quantity * rate

If a numeric field is still unknown after this reasoning, set it to 0.0.

OUTPUT FORMAT (PER BATCH):

You will see K page images in this batch, in order.
For EACH page i in this batch (1-based):

- Decide page_type ∈ {"Bill Detail", "Final Bill", "Pharmacy"}.
- Extract ALL line items for that page (one per visual row).

Return ONE STRICT JSON object ONLY (no markdown):

{
  "pagewise_line_items": [
    {
      "page_no": "1",
      "page_type": "Bill Detail" | "Final Bill" | "Pharmacy",
      "bill_items": [
        {
          "item_name": "<string>",
          "item_amount": <float>,
          "item_rate": <float>,
          "item_quantity": <float>
        }
      ]
    }
  ],
  "total_item_count": <integer>
}

- page_no is the 1-based index WITHIN THIS BATCH, as a STRING.
- bill_items may be [] for pages with no charges.
- total_item_count is the number of bill_items across all pages in the batch.

STRICT REQUIREMENTS:
- JSON ONLY. No ```json, no headings, no commentary.
- No extra keys at top level or inside any object.
- Do NOT output totals, sub-totals, taxes, or summary-only rows as bill_items.
"""

SYSTEM_PROMPT_REFINEMENT = """
You are a precise hospital BILL ITEM extraction engine.

You will see ONLY ONE page image of a hospital bill.

Task:

- Read all charge tables on this page.
- For EVERY visible row that represents a real charge (with description + amount),
  output ONE bill_items entry.
- Do NOT output totals, sub-totals, taxes, or purely summary rows.
- Do NOT add any commentary. Return JSON ONLY.

Output format for this SINGLE PAGE:

{
  "pagewise_line_items": [
    {
      "page_no": "1",
      "page_type": "Bill Detail" | "Final Bill" | "Pharmacy",
      "bill_items": [
        {
          "item_name": "<string>",
          "item_amount": <float>,
          "item_rate": <float>,
          "item_quantity": <float>
        }
      ]
    }
  ],
  "total_item_count": <integer>
}

Rules:
- JSON only (no markdown, no comments).
- Numeric fields must be numbers, not strings.
- If quantity is missing but amount is visible: quantity=1.0, rate=amount.
- If rate is missing but quantity & amount are visible: rate=amount/quantity.
- If a numeric field is unreadable: set 0.0.
"""

# ============================================================
#  LLM Call Helpers
# ============================================================

def parse_llm_json(raw_text: str, src: str) -> Any:
    """
    Robust JSON parsing: strip markdown fences, take the outermost {...}.
    Raises HTTPException if it fails.
    """
    text = raw_text.strip()

    # Remove ```xxx fences if present
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 3:
            text = "".join(parts[1:-1]).strip()
            if "\n" in text:
                first_line, rest = text.split("\n", 1)
                if first_line.strip().lower() in ("json", "javascript"):
                    text = rest.strip()

    # Now try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last != -1 and last > first:
            json_str = text[first: last + 1]
            try:
                return json.loads(json_str)
            except Exception:
                pass
        raise HTTPException(
            status_code=500,
            detail=f"{src} response is not valid JSON: {text[:200]}",
        )


def call_groq_for_batch(
    batch_page_infos: List[Dict[str, Any]],
    model_id: str,
) -> Tuple[List[Dict[str, Any]], TokenUsage]:
    """
    Single Groq Vision call for a batch of pages (size <= MAX_IMAGES_PER_REQUEST).
    Returns:
        - raw_pages: list of per-page dicts (page_no, page_type, bill_items)
        - token_usage: TokenUsage for this batch
    """
    num_batch_pages = len(batch_page_infos)

    user_text = f"""
You are given a BATCH of {num_batch_pages} page image(s) from a hospital bill.

The images will be provided in order for this batch:
first image is batch page 1, second is batch page 2, ..., up to batch page {num_batch_pages}.

For EACH batch page i (1-based), you must:
- Set page_no = "<i>" (as a string)
- Choose page_type from: "Bill Detail", "Final Bill", "Pharmacy"
- Extract bill_items ONLY for that page.
"""

    user_content: List[Dict[str, Any]] = [
        {"type": "input_text", "text": user_text.strip()}
    ]

    # Use lower detail for Scout to save time, higher for Maverick
    detail_level = "high" if "maverick" in model_id else "auto"

    for idx, info in enumerate(batch_page_infos):
        batch_page_no = idx + 1
        user_content.append(
            {
                "type": "input_text",
                "text": f"BATCH PAGE {batch_page_no} IMAGE BELOW.",
            }
        )
        user_content.append(
            {
                "type": "input_image",
                "image_url": info["data_url"],
                "detail": detail_level,
            }
        )

    try:
        response = client.responses.create(
            model=model_id,
            input=[
                {
                    "role": "system",
                    "content": [
                        {"type": "input_text", "text": SYSTEM_PROMPT_BULK.strip()}
                    ],
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
        )
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Groq API error (bulk batch): {e}",
        )

    usage = getattr(response, "usage", None)
    if usage is not None:
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    else:
        total_tokens = input_tokens = output_tokens = 0

    token_usage = TokenUsage(
        total_tokens=total_tokens,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    raw_text = response.output_text
    parsed = parse_llm_json(raw_text, src="Bulk model")

    if isinstance(parsed, dict) and "pagewise_line_items" in parsed:
        raw_pages = parsed.get("pagewise_line_items", []) or []
    elif isinstance(parsed, list):
        raw_pages = parsed
    else:
        raise HTTPException(
            status_code=500,
            detail="Bulk JSON does not contain 'pagewise_line_items' list.",
        )

    # Guarantee one page object per input page in this batch
    if len(raw_pages) < num_batch_pages:
        for idx in range(len(raw_pages), num_batch_pages):
            raw_pages.append(
                {
                    "page_no": str(idx + 1),
                    "page_type": "Bill Detail",
                    "bill_items": [],
                }
            )

    raw_pages = raw_pages[:num_batch_pages]
    return raw_pages, token_usage


def call_groq_for_single_page_refine(
    page_info: Dict[str, Any],
    model_id: str,
) -> Tuple[Dict[str, Any], TokenUsage]:
    """
    Single-page refinement using Maverick (or any precise model).
    Returns:
        - raw_page: one dict with page_no="1", page_type, bill_items
        - token_usage: TokenUsage for this call
    """
    user_content: List[Dict[str, Any]] = [
        {
            "type": "input_text",
            "text": "This is a single hospital bill page. Extract all charge line items only.",
        },
        {
            "type": "input_image",
            "image_url": page_info["data_url"],
            "detail": "high",
        },
    ]

    try:
        response = client.responses.create(
            model=model_id,
            input=[
                {
                    "role": "system",
                    "content": [
                        {"type": "input_text", "text": SYSTEM_PROMPT_REFINEMENT.strip()}
                    ],
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
        )
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Groq API error (refine single-page): {e}",
        )

    usage = getattr(response, "usage", None)
    if usage is not None:
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    else:
        total_tokens = input_tokens = output_tokens = 0

    token_usage = TokenUsage(
        total_tokens=total_tokens,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    raw_text = response.output_text
    parsed = parse_llm_json(raw_text, src="Refine model")

    if isinstance(parsed, dict) and "pagewise_line_items" in parsed:
        raw_pages = parsed.get("pagewise_line_items", []) or []
    elif isinstance(parsed, list):
        raw_pages = parsed
    else:
        raise HTTPException(
            status_code=500,
            detail="Refine JSON does not contain 'pagewise_line_items' list.",
        )

    if not raw_pages:
        raise HTTPException(
            status_code=500,
            detail="Refine JSON returned empty pagewise_line_items.",
        )

    raw_page = raw_pages[0]
    return raw_page, token_usage


# ============================================================
#  Reconcile, Clean & Aggregate
# ============================================================

def _coerce_number(x: Any) -> float:
    """Coerce messy numeric value (None, '', '—', etc.) to float."""
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


def clean_page_dict(page_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pre-clean the raw JSON dict from the model so that Pydantic parsing will not
    fail when numeric fields are null/empty/etc.

    IMPORTANT: We DO NOT dedupe within a page here – multiple identical rows
    (e.g. many IP CONSULTATION CHARGES) must be preserved as separate items.
    """
    bill_items = page_dict.get("bill_items", []) or []
    cleaned_items: List[Dict[str, Any]] = []

    for item in bill_items:
        if not isinstance(item, dict):
            continue

        name = str(item.get("item_name", "")).strip()
        amount = _coerce_number(item.get("item_amount"))
        rate = _coerce_number(item.get("item_rate"))
        qty = _coerce_number(item.get("item_quantity"))

        cleaned_items.append(
            {
                "item_name": name,
                "item_amount": amount,
                "item_rate": rate,
                "item_quantity": qty,
            }
        )

    return {
        "page_no": str(page_dict.get("page_no", "")),
        "page_type": str(page_dict.get("page_type", "Bill Detail")),
        "bill_items": cleaned_items,
    }


def reconcile_page_items(page_dict: Dict[str, Any]) -> PageItems:
    """
    Validate and normalize a single page's JSON dict into PageItems.
    Fix small inconsistencies between rate, qty and amount.
    """
    cleaned = clean_page_dict(page_dict)

    try:
        page_items = PageItems(**cleaned)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model JSON does not match expected schema: {e}",
        )

    EPS = 0.01
    for item in page_items.bill_items:
        amount = float(item.item_amount)
        rate = float(item.item_rate)
        qty = float(item.item_quantity)

        if rate and qty:
            computed = rate * qty
            if math.isfinite(computed) and abs(computed - amount) > EPS:
                item.item_amount = round(computed, 2)
        elif amount and qty and qty != 0:
            item.item_rate = round(amount / qty, 4)
        elif amount and (not qty or qty == 0):
            item.item_quantity = 1.0
            item.item_rate = round(amount, 2)

    return page_items


def enrich_from_patterns(pages: List[PageItems]) -> List[PageItems]:
    """
    SECOND PASS:
    If some rows have missing numeric fields, but other rows with the same
    item_name have good numbers, use that pattern to fill in.
    """
    rates = defaultdict(list)
    qtys = defaultdict(list)

    for p in pages:
        for it in p.bill_items:
            name_key = it.item_name.strip().lower()
            amt = float(it.item_amount)
            rate = float(it.item_rate)
            qty = float(it.item_quantity)

            if rate > 0 and qty > 0:
                rates[name_key].append(rate)
                qtys[name_key].append(qty)
            elif amt > 0 and qty > 0:
                rates[name_key].append(amt / qty)
                qtys[name_key].append(qty)

    default_rate = {k: (sum(v) / len(v)) for k, v in rates.items() if v}
    default_qty = {k: (sum(v) / len(v)) for k, v in qtys.items() if v}

    for p in pages:
        for it in p.bill_items:
            name_key = it.item_name.strip().lower()
            amt = float(it.item_amount)
            rate = float(it.item_rate)
            qty = float(it.item_quantity)

            dr = default_rate.get(name_key)
            dq = default_qty.get(name_key, 1.0)

            if dr is not None:
                if qty <= 0:
                    qty = dq or 1.0
                if rate <= 0 and amt > 0 and qty > 0:
                    rate = amt / qty
                if rate <= 0 and amt <= 0:
                    rate = dr
                if amt <= 0 and rate > 0 and qty > 0:
                    amt = rate * qty

            it.item_quantity = float(qty if qty > 0 else 1.0)
            it.item_rate = round(float(rate), 2) if rate > 0 else 0.0
            it.item_amount = round(float(amt), 2) if amt > 0 else 0.0

    return pages


def aggregate_all_pages(pages: List[PageItems]) -> ExtractBillDataResponseData:
    total_items = sum(len(p.bill_items) for p in pages)
    return ExtractBillDataResponseData(
        pagewise_line_items=pages,
        total_item_count=total_items,
    )


def compute_grand_total_amount(pages: List[PageItems]) -> float:
    total = 0.0
    for p in pages:
        for item in p.bill_items:
            total += float(item.item_amount)
    return round(total, 2)


# ============================================================
#  Health Check (GET)
# ============================================================

@app.get("/extract-bill-data")
def health_check():
    return {
        "message": "Health OK. Use POST /extract-bill-data with JSON body "
        '{"document": "<public image/PDF URL>"} to extract bill data.'
    }


# ============================================================
#  Main Datathon Endpoint (POST)
# ============================================================

@app.post("/extract-bill-data", response_model=ExtractBillDataResponse)
def extract_bill_data(req: ExtractBillDataRequest):
    start_time = time.time()
    url_str = str(req.document)

    # 1. Download document
    content = download_document(url_str)

    # 2. Load RAW pages
    raw_pages = load_document_pages(url_str, content)
    num_pages = len(raw_pages)

    if num_pages == 0:
        elapsed = time.time() - start_time
        print(
            f"[BILL_EXTRACT] pages=0 items=0 total_amount=0.00 tokens=0 "
            f"time_sec={elapsed:.2f} (no pages)"
        )
        return ExtractBillDataResponse(
            is_success=False,
            message="No pages/images could be extracted from the document.",
        )

    # 3. Choose dynamic strategy & build page_infos
    strategy = choose_strategy(num_pages)
    page_infos = build_page_infos(raw_pages, strategy)

    # 4. Bulk pass – process pages in batches
    all_pages: List[PageItems] = []
    total_tokens = 0
    input_tokens = 0
    output_tokens = 0

    batch_size = strategy["bulk_batch_size"]

    for batch_start in range(0, num_pages, batch_size):
        batch_end = min(batch_start + batch_size, num_pages)
        batch_page_infos = page_infos[batch_start:batch_end]

        raw_pages_batch, usage_batch = call_groq_for_batch(
            batch_page_infos,
            model_id=strategy["bulk_model"],
        )

        total_tokens += usage_batch.total_tokens
        input_tokens += usage_batch.input_tokens
        output_tokens += usage_batch.output_tokens

        # Map batch-local page_no → global page_no
        for i, page_dict in enumerate(raw_pages_batch):
            global_page_no = batch_start + i + 1  # 1-based for entire document
            page_dict["page_no"] = str(global_page_no)
            page_items = reconcile_page_items(page_dict)
            all_pages.append(page_items)

    if not all_pages:
        elapsed = time.time() - start_time
        print(
            f"[BILL_EXTRACT] pages={num_pages} items=0 total_amount=0.00 "
            f"tokens={total_tokens} time_sec={elapsed:.2f} (no items)"
        )
        return ExtractBillDataResponse(
            is_success=False,
            token_usage=TokenUsage(
                total_tokens=total_tokens,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
            message="Model did not return any page items.",
        )

    # 5. Optional refinement with Maverick on suspicious pages
    if strategy.get("use_refine", True) and strategy.get("refine_limit", 0) > 0:
        suspicious_indices: List[int] = []
        for idx, p in enumerate(all_pages):
            ocr_rows = page_infos[idx].get("ocr_rows", 0)
            n_items = len(p.bill_items)
            page_type = (p.page_type or "").strip().lower()

            # Heuristics:
            # - Very sparse vs OCR row count
            # - Final bill pages
            # - Almost empty but OCR shows content
            sparse_vs_ocr = (
                ocr_rows >= 8 and n_items <= max(3, ocr_rows // 2)
            )
            almost_empty = (n_items <= 1 and ocr_rows >= 5)
            suspicious_final = (page_type == "final bill" and n_items < 5)

            if sparse_vs_ocr or almost_empty or suspicious_final:
                suspicious_indices.append(idx)

        suspicious_indices = suspicious_indices[: strategy["refine_limit"]]

        for idx in suspicious_indices:
            try:
                raw_page_ref, usage_ref = call_groq_for_single_page_refine(
                    page_infos[idx],
                    model_id=strategy["refine_model"],
                )
            except HTTPException as e:
                print(f"[REFINE_SKIP] page={idx+1} reason={e.detail}")
                continue
            except Exception as e:
                print(f"[REFINE_SKIP] page={idx+1} unexpected_error={e}")
                continue

            total_tokens += usage_ref.total_tokens
            input_tokens += usage_ref.input_tokens
            output_tokens += usage_ref.output_tokens

            raw_page_ref["page_no"] = str(idx + 1)
            refined_page = reconcile_page_items(raw_page_ref)
            all_pages[idx] = refined_page

    # 6. Enrich missing numeric fields using global patterns
    all_pages = enrich_from_patterns(all_pages)

    # 7. Aggregate
    data = aggregate_all_pages(all_pages)

    token_usage = TokenUsage(
        total_tokens=total_tokens,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    # 8. Logging (simple, no extra accuracy logs)
    grand_total = compute_grand_total_amount(all_pages)
    elapsed = time.time() - start_time

    print(
        f"[BILL_EXTRACT] pages={num_pages} "
        f"items={data.total_item_count} "
        f"total_amount={grand_total:.2f} "
        f"tokens={token_usage.total_tokens} "
        f"time_sec={elapsed:.2f}"
    )

    return ExtractBillDataResponse(
        is_success=True,
        token_usage=token_usage,
        data=data,
    )
