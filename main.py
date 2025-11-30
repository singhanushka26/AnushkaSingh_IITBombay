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

# Optional OCR (pytesseract)
try:
    import pytesseract
    HAS_PYTESSERACT = True
except ImportError:
    pytesseract = None
    HAS_PYTESSERACT = False

# ============================================================
#  Groq Client (OpenAI-compatible)
# ============================================================

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

# Fast model for bulk extraction
GROQ_VISION_MODEL_SCOUT = os.environ.get(
    "GROQ_VISION_MODEL_SCOUT",
    "meta-llama/llama-4-scout-17b-16e-instruct",
)

# More accurate model for refinement
GROQ_VISION_MODEL_MAVERICK = os.environ.get(
    "GROQ_VISION_MODEL_MAVERICK",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
)

# Groq Vision limit: MAX 5 images per request
MAX_IMAGES_PER_REQUEST = 5

# Hard time budget (seconds) – we try to stay roughly under this
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
    version="8.0.0",
    description=(
        "Extracts line items from multi-page bill/invoice documents using "
        "Groq vision models with batching (max 5 images per request). "
        "Scout does bulk extraction; Maverick refines suspicious pages with OCR hints. "
        "Implements the exact HackRx Datathon schema and is tuned for "
        "high accuracy while staying under time limits on large PDFs."
    ),
)


# ============================================================
#  Helpers – Download & Convert Documents
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


def _resize_for_vision(img: Image.Image, max_dim: int = 950) -> Image.Image:
    """
    Downscale to reduce tokens & latency while keeping table details readable.
    Slightly smaller than before for better speed on 20–30 pages.
    """
    w, h = img.size
    scale = max(w, h) / float(max_dim)
    if scale <= 1.0:
        return img
    new_w = int(w / scale)
    new_h = int(h / scale)
    return img.resize((new_w, new_h), Image.LANCZOS)


def _enhance_for_vision(img: Image.Image) -> Image.Image:
    """
    Light contrast + sharpness boost to help faint numeric columns.
    """
    img = ImageEnhance.Contrast(img).enhance(1.6)
    img = ImageEnhance.Sharpness(img).enhance(1.4)
    return img


def pil_to_data_url(img: Image.Image, quality: int = 60) -> str:
    """
    Convert a PIL Image into a base64 JPEG data URL.
    Assumes img is already reasonably sized.
    """
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b = buf.getvalue()

    # Ensure < 4 MB per image
    q = quality
    while len(b) > 4 * 1024 * 1024 and q > 30:
        q -= 10
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        b = buf.getvalue()

    b64 = base64.b64encode(b).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def image_to_data_url(img: Image.Image, quality: int = 60) -> str:
    """
    Convert a PIL Image into a base64 JPEG data URL, with resize + enhance.
    """
    img = _resize_for_vision(img, max_dim=950)
    img = _enhance_for_vision(img)
    return pil_to_data_url(img, quality=quality)


def data_url_to_pil(data_url: str) -> Image.Image:
    """
    Decode a data URL back into a PIL Image.
    Used for cropping + OCR in refinement.
    """
    try:
        header, b64data = data_url.split(",", 1)
    except ValueError:
        b64data = data_url
    img_bytes = base64.b64decode(b64data)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return img


def document_to_page_infos(url: str, content: bytes) -> List[Dict[str, Any]]:
    """
    Convert the downloaded document into a list of PAGE INFOS:

        {
          "page_index": int,   # 0-based
          "data_url": "data:image/jpeg;base64,..."
        }
    """
    mime = guess_mime_type(url, content)
    page_infos: List[Dict[str, Any]] = []

    # Single images
    if mime.startswith("image/"):
        try:
            img = Image.open(io.BytesIO(content)).convert("RGB")
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Unable to open image document: {e}",
            )

        page_infos.append(
            {
                "page_index": 0,
                "data_url": image_to_data_url(img),
            }
        )
        return page_infos

    # PDFs
    if mime == "application/pdf":
        try:
            pages = convert_from_bytes(content)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Unable to convert PDF to images. "
                    "Ensure poppler is installed and available in PATH. "
                    f"Error: {e}"
                ),
            )

        for idx, p in enumerate(pages):
            img = p.convert("RGB")
            page_infos.append(
                {
                    "page_index": idx,
                    "data_url": image_to_data_url(img),
                }
            )
        return page_infos

    # Fallback: try as image anyway
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported document type: {mime}",
        )

    page_infos.append(
        {
            "page_index": 0,
            "data_url": image_to_data_url(img),
        }
    )
    return page_infos


# ============================================================
#  LLM Prompts – Scout (batch) + Maverick (single page)
# ============================================================

SYSTEM_PROMPT_SCOUT = """
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

SYSTEM_PROMPT_MAVERICK = """
You are a precise hospital BILL ITEM extraction engine.

You will see ONE hospital bill page (with an optional zoomed crop).
You may also see OCR text for that page.

Your task:

- Read all charge tables on this page.
- For EVERY visible row that represents a real charge (with description + amount),
  output ONE bill_items entry.
- DO NOT output totals, sub-totals, purely summary/tax rows, or grand totals.
- JSON ONLY – no markdown, no comments.

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
- Numeric fields must be numbers, not strings.
- If quantity is missing but amount is visible: quantity = 1.0, rate = amount.
- If rate is missing but quantity & amount are visible: rate = amount / quantity.
- If a numeric field is unreadable: set it to 0.0.
"""


# ============================================================
#  Groq Calls – Scout batch + Maverick single-page
# ============================================================

def _safe_json_loads(raw_text: str, context: str) -> Any:
    """
    Robust JSON parsing:
    - First try json.loads directly
    - If that fails, strip everything before first '{' and after last '}'
    """
    raw_text = raw_text.strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        first = raw_text.find("{")
        last = raw_text.rfind("}")
        if first != -1 and last != -1 and last > first:
            json_str = raw_text[first:last + 1]
            try:
                return json.loads(json_str)
            except Exception:
                raise HTTPException(
                    status_code=500,
                    detail=f"{context} response is not valid JSON (inner slice): {raw_text[:200]}",
                )
        raise HTTPException(
            status_code=500,
            detail=f"{context} response is not valid JSON (no braces): {raw_text[:200]}",
        )


def call_groq_for_batch(
    batch_page_infos: List[Dict[str, Any]],
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
                "detail": "auto",
            }
        )

    try:
        response = client.responses.create(
            model=GROQ_VISION_MODEL_SCOUT,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": SYSTEM_PROMPT_SCOUT.strip()}],
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
            detail=f"Groq API error (Scout batch): {e}",
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

    raw_text = response.output_text.strip()
    parsed = _safe_json_loads(raw_text, context="Scout")

    if isinstance(parsed, dict) and "pagewise_line_items" in parsed:
        raw_pages = parsed.get("pagewise_line_items", []) or []
    elif isinstance(parsed, list):
        raw_pages = parsed
    else:
        raise HTTPException(
            status_code=500,
            detail="Scout JSON for batch does not contain 'pagewise_line_items' list.",
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


def get_ocr_hint_for_page(page_info: Dict[str, Any]) -> str:
    """
    Use pytesseract to get a short OCR hint string for this page.
    If pytesseract is not available, returns empty string.
    """
    if not HAS_PYTESSERACT:
        return ""

    try:
        img = data_url_to_pil(page_info["data_url"])
    except Exception:
        return ""

    try:
        text = pytesseract.image_to_string(img, config="--psm 6")
    except Exception:
        return ""

    # Keep only first ~800 chars to control token usage
    text = text.strip().replace("\n\n", "\n")
    if len(text) > 800:
        text = text[:800]

    return text


def get_zoom_crop_data_url(page_info: Dict[str, Any]) -> Optional[str]:
    """
    Create a zoomed crop of the central band of the page where tables usually are.
    Returns a data_url for the crop, or None on failure.
    """
    try:
        img = data_url_to_pil(page_info["data_url"])
    except Exception:
        return None

    w, h = img.size
    # Central vertical band: from 18% to 88% height, with small horizontal margins
    left = int(0.05 * w)
    right = int(0.95 * w)
    top = int(0.18 * h)
    bottom = int(0.88 * h)

    if right <= left or bottom <= top:
        return None

    crop = img.crop((left, top, right, bottom))
    crop = _resize_for_vision(crop, max_dim=900)
    crop = _enhance_for_vision(crop)
    return pil_to_data_url(crop, quality=65)


def call_groq_for_single_page_maverick(
    page_info: Dict[str, Any],
    use_ocr: bool = True,
    use_crop: bool = True,
) -> Tuple[Dict[str, Any], TokenUsage]:
    """
    Single-page refinement using Maverick with optional crop + OCR.
    Returns:
        - raw_page: one dict with page_no="1", page_type, bill_items
        - token_usage: TokenUsage for this Maverick call

    Any JSON/LLM error is raised and handled by caller (fallback to Scout).
    """
    user_content: List[Dict[str, Any]] = []

    # Optional OCR hint
    if use_ocr:
        ocr_hint = get_ocr_hint_for_page(page_info)
        if ocr_hint:
            user_content.append(
                {
                    "type": "input_text",
                    "text": (
                        "OCR text from this page (may contain noise, use it only to "
                        "understand column names and dates):\n" + ocr_hint
                    ),
                }
            )

    # Explain image ordering
    if use_crop:
        user_content.append(
            {
                "type": "input_text",
                "text": "First image is a zoomed crop of the main charge tables; second is the full page.",
            }
        )
        crop_url = get_zoom_crop_data_url(page_info)
        if crop_url:
            user_content.append(
                {
                    "type": "input_image",
                    "image_url": crop_url,
                    "detail": "high",
                }
            )
        # Full page
        user_content.append(
            {
                "type": "input_image",
                "image_url": page_info["data_url"],
                "detail": "high",
            }
        )
    else:
        user_content.append(
            {
                "type": "input_image",
                "image_url": page_info["data_url"],
                "detail": "high",
            }
        )

    try:
        response = client.responses.create(
            model=GROQ_VISION_MODEL_MAVERICK,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": SYSTEM_PROMPT_MAVERICK.strip()}],
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
            detail=f"Groq API error (Maverick single-page): {e}",
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

    raw_text = response.output_text.strip()
    parsed = _safe_json_loads(raw_text, context="Maverick")

    if isinstance(parsed, dict) and "pagewise_line_items" in parsed:
        raw_pages = parsed.get("pagewise_line_items", []) or []
    elif isinstance(parsed, list):
        raw_pages = parsed
    else:
        raise HTTPException(
            status_code=500,
            detail="Maverick JSON does not contain 'pagewise_line_items' list.",
        )

    if not raw_pages:
        raise HTTPException(
            status_code=500,
            detail="Maverick JSON returned empty pagewise_line_items.",
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
    rates = defaultdict(list)  # name -> list of inferred rates
    qtys = defaultdict(list)   # name -> list of typical quantities

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


def compute_maverick_budget(num_pages: int) -> int:
    """
    Dynamic budget: fewer pages → more Maverick refinement.
    This is a heuristic to stay under ~120s total time.
    """
    if num_pages <= 5:
        return min(num_pages, 5)      # Almost all pages
    if num_pages <= 10:
        return min(num_pages, 7)
    if num_pages <= 15:
        return min(num_pages, 6)
    if num_pages <= 20:
        return min(num_pages, 5)
    if num_pages <= 25:
        return min(num_pages, 4)
    return min(num_pages, 3)          # Very large docs, small refinement


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

    # 2. Convert to per-page infos (image → data_url)
    page_infos = document_to_page_infos(url_str, content)
    num_pages = len(page_infos)

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

    # 3. Scout pass – process pages in batches of <= MAX_IMAGES_PER_REQUEST
    all_pages: List[PageItems] = []
    total_tokens = 0
    input_tokens = 0
    output_tokens = 0

    for batch_start in range(0, num_pages, MAX_IMAGES_PER_REQUEST):
        batch_end = min(batch_start + MAX_IMAGES_PER_REQUEST, num_pages)
        batch_page_infos = page_infos[batch_start:batch_end]

        raw_pages_batch, usage_batch = call_groq_for_batch(batch_page_infos)

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

    # 4. Decide Maverick refinement budget dynamically
    #    We refine "suspicious" pages up to this budget.
    max_maverick_pages = compute_maverick_budget(num_pages)

    # Rank suspicious pages: few items or "Final Bill" pages first
    suspicious_indices: List[int] = []
    for idx, p in enumerate(all_pages):
        ptype = p.page_type.strip().lower()
        num_items = len(p.bill_items)
        score = 0
        if ptype == "final bill":
            score += 100
        if num_items <= 1:
            score += 50
        elif num_items <= 3:
            score += 25
        if score > 0:
            suspicious_indices.append((score, idx))

    # Sort by suspiciousness descending
    suspicious_indices.sort(key=lambda x: x[0], reverse=True)
    selected_indices = [idx for _, idx in suspicious_indices][:max_maverick_pages]

    # 5. Maverick refinement on selected pages (best-effort, never crashes)
    for idx in selected_indices:
        try:
            raw_page_mav, usage_mav = call_groq_for_single_page_maverick(
                page_infos[idx],
                use_ocr=True,
                use_crop=True,
            )
        except HTTPException as e:
            print(f"[MAVERICK_SKIP] page={idx+1} reason={e.detail}")
            continue
        except Exception as e:
            print(f"[MAVERICK_SKIP] page={idx+1} unexpected_error={e}")
            continue

        total_tokens += usage_mav.total_tokens
        input_tokens += usage_mav.input_tokens
        output_tokens += usage_mav.output_tokens

        # Map Maverick's local page_no ("1") to global page number
        raw_page_mav["page_no"] = str(idx + 1)
        refined_page = reconcile_page_items(raw_page_mav)
        all_pages[idx] = refined_page

        # Optional early stop if we are close to global time budget
        elapsed_mid = time.time() - start_time
        if elapsed_mid > GLOBAL_TIME_BUDGET_SEC * 0.9:
            print(
                f"[MAVERICK_EARLY_STOP] elapsed={elapsed_mid:.2f}s, "
                f"stopping further refinement to respect time budget."
            )
            break

    # 6. Enrich missing numeric fields using global patterns
    all_pages = enrich_from_patterns(all_pages)

    # 7. Aggregate
    data = aggregate_all_pages(all_pages)

    token_usage = TokenUsage(
        total_tokens=total_tokens,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    # 8. Logging
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
