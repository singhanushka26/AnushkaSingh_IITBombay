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

# Optional OCR (not used in this Max-Accuracy version, kept for future tuning)
try:
    import pytesseract  # noqa: F401

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

# Fast bulk model (Scout)
GROQ_VISION_MODEL_SCOUT = os.environ.get(
    "GROQ_VISION_MODEL_SCOUT",
    "meta-llama/llama-4-scout-17b-16e-instruct",
)

# Accurate model (Maverick) – used on ALL pages in this mode
GROQ_VISION_MODEL_MAVERICK = os.environ.get(
    "GROQ_VISION_MODEL_MAVERICK",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
)

# Groq Vision limit: MAX 5 images per request
MAX_IMAGES_PER_REQUEST = 5

# Soft global budget (conceptual, not enforced)
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
    version="10.0.0-hybrid-max-accuracy",
    description=(
        "HYBRID MAX-ACCURACY: "
        "Scout bulk pass + full Maverick pass on all pages, "
        "with deep merge and numeric integrity repair. "
        "Optimized for highest possible accuracy under ~120s."
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

    This is deliberately conservative:
    - Crop ~14% top, ~8% bottom, ~4% left/right.
    """
    w, h = img.size
    top = int(0.14 * h)
    bottom = int(0.92 * h)
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
#  Strategy Selection – Scout + Maverick
# ============================================================

def choose_scout_strategy(num_pages: int) -> Dict[str, Any]:
    """
    Strategy for the Scout bulk pass.

    - Medium resolution & compression.
    - Batch size ~3 for most docs.
    """
    strategy: Dict[str, Any] = {
        "model": GROQ_VISION_MODEL_SCOUT,
        "batch_size": 3,
        "max_dim": 850,
        "contrast": 1.4,
        "sharpness": 1.3,
        "jpeg_quality": 45,
        "use_smart_crop": True,
    }

    if num_pages <= 4:
        strategy["batch_size"] = 2
        strategy["max_dim"] = 900
        strategy["jpeg_quality"] = 50
    elif num_pages <= 10:
        strategy["batch_size"] = 3
        strategy["max_dim"] = 850
        strategy["jpeg_quality"] = 45
    elif num_pages <= 20:
        strategy["batch_size"] = 3
        strategy["max_dim"] = 820
        strategy["jpeg_quality"] = 42
    else:
        strategy["batch_size"] = 4
        strategy["max_dim"] = 800
        strategy["jpeg_quality"] = 40

    strategy["batch_size"] = min(strategy["batch_size"], MAX_IMAGES_PER_REQUEST)
    return strategy


def choose_maverick_strategy(num_pages: int) -> Dict[str, Any]:
    """
    Strategy for the Maverick high-accuracy pass.

    R1 + M2:
    - max_dim = 1100 (high resolution)
    - batch_size = 2
    - slightly higher JPEG quality.
    """
    strategy: Dict[str, Any] = {
        "model": GROQ_VISION_MODEL_MAVERICK,
        "batch_size": 2,   # M2
        "max_dim": 1100,   # R1
        "contrast": 1.5,
        "sharpness": 1.4,
        "jpeg_quality": 55,
        "use_smart_crop": True,
    }

    # Very large documents: slightly tighten for safety
    if num_pages > 20:
        strategy["max_dim"] = 1000
        strategy["jpeg_quality"] = 50

    strategy["batch_size"] = min(strategy["batch_size"], MAX_IMAGES_PER_REQUEST)
    return strategy


def build_page_infos(
    raw_pages: List[Image.Image],
    max_dim: int,
    contrast: float,
    sharpness: float,
    jpeg_quality: int,
    use_smart_crop: bool = True,
) -> List[Dict[str, Any]]:
    """
    Apply cropping, enhancement, resizing and JPEG encoding.

    Returns:
        List[Dict]: each entry has:
            - page_index: int
            - data_url: str
    """
    page_infos: List[Dict[str, Any]] = []
    for idx, img in enumerate(raw_pages):
        img_proc = img.convert("RGB")
        if use_smart_crop:
            img_proc = smart_crop(img_proc)
        img_proc = enhance_image(img_proc, contrast_factor=contrast, sharpness_factor=sharpness)
        img_proc = resize_image_max_dim(img_proc, max_dim=max_dim)

        jpeg_bytes = image_to_jpeg_bytes(img_proc, quality=jpeg_quality)
        data_url = jpeg_bytes_to_data_url(jpeg_bytes)

        page_infos.append(
            {
                "page_index": idx,
                "data_url": data_url,
            }
        )
    return page_infos


# ============================================================
#  LLM Prompts – Compact BULK prompt (used for both Scout & Maverick)
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


# ============================================================
#  Response Text Helper
# ============================================================

def extract_text_from_response(response: Any) -> str:
    """
    Groq Responses helper.

    Prefer response.output_text if available; otherwise attempt to
    pull text from response.output[0].content[0].text.
    """
    if hasattr(response, "output_text"):
        return response.output_text

    try:
        output_blocks = getattr(response, "output", None)
        if output_blocks and len(output_blocks) > 0:
            first_block = output_blocks[0]
            content = getattr(first_block, "content", None)
            if content and len(content) > 0:
                first_piece = content[0]
                text = getattr(first_piece, "text", None)
                if isinstance(text, str):
                    return text
    except Exception:
        pass

    raise HTTPException(
        status_code=500,
        detail="LLM response does not contain text output.",
    )


# ============================================================
#  JSON Parsing with Self-Repair
# ============================================================

def parse_llm_json(raw_text: str, src: str) -> Any:
    """
    Robust JSON parsing with light self-repair:

    - Strips ``` fences and language tags.
    - Extracts outermost {...}.
    - Replaces common non-JSON tokens:
        NaN, Infinity, -Infinity → 0
    - Fixes simple trailing comma patterns: ",]" → "]", ",}" → "}".
    """
    text = raw_text.strip()

    # Remove ``` fences if present
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 3:
            text = "".join(parts[1:-1]).strip()
            if "\n" in text:
                first_line, rest = text.split("\n", 1)
                if first_line.strip().lower() in ("json", "javascript"):
                    text = rest.strip()

    # Always try to take outermost {...}
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        text = text[first : last + 1]

    # Light sanitization for common non-JSON tokens
    sanitized = text
    sanitized = sanitized.replace("NaN", "0")
    sanitized = sanitized.replace("Infinity", "0")
    sanitized = sanitized.replace("-Infinity", "0")

    # Remove trailing commas before lists/objects close
    sanitized = sanitized.replace(",]", "]")
    sanitized = sanitized.replace(", ]", "]")
    sanitized = sanitized.replace(",}", "}")
    sanitized = sanitized.replace(", }", "}")

    # First attempt
    try:
        return json.loads(sanitized)
    except json.JSONDecodeError:
        # Last fallback: try again on the tightest outer braces
        first = sanitized.find("{")
        last = sanitized.rfind("}")
        if first != -1 and last != -1 and last > first:
            json_str = sanitized[first : last + 1]
            try:
                return json.loads(json_str)
            except Exception:
                pass

        snippet = sanitized[:200].replace("\n", " ")
        raise HTTPException(
            status_code=500,
            detail=f"{src} response is not valid JSON: {snippet}",
        )


# ============================================================
#  LLM Call Helper – Generic Batch (Scout & Maverick)
# ============================================================

def call_groq_for_batch(
    batch_page_infos: List[Dict[str, Any]],
    model_id: str,
    image_detail: str = "auto",
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
- Set page_no = "<i>" (as a string).
- Choose page_type from: "Bill Detail", "Final Bill", "Pharmacy".
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
                "detail": image_detail,
            }
        )

    try:
        response = client.responses.create(
            model=model_id,
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": SYSTEM_PROMPT_BULK.strip(),
                        }
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
            detail=f"Groq API error (batch, model={model_id}): {e}",
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

    raw_text = extract_text_from_response(response)
    parsed = parse_llm_json(raw_text, src=f"Model {model_id}")

    if isinstance(parsed, dict) and "pagewise_line_items" in parsed:
        raw_pages = parsed.get("pagewise_line_items", []) or []
    elif isinstance(parsed, list):
        raw_pages = parsed
    else:
        raise HTTPException(
            status_code=500,
            detail=f"JSON from model {model_id} does not contain 'pagewise_line_items' list.",
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


def merge_scout_and_maverick_pages(
    scout_pages: List[PageItems],
    mav_pages: List[PageItems],
) -> List[PageItems]:
    """
    Merge per-page results from Scout and Maverick.

    Rules:
    - If Maverick has any rows → use Maverick as base for that page.
    - Keep all Maverick rows.
    - Add any Scout rows that are not duplicates of Maverick rows.
    - Prefer Maverick page_type if available, else Scout.
    """
    merged: List[PageItems] = []
    n = max(len(scout_pages), len(mav_pages))

    for i in range(n):
        s_page = scout_pages[i] if i < len(scout_pages) else None
        m_page = mav_pages[i] if i < len(mav_pages) else None

        # If Maverick output is missing, fall back to Scout
        if m_page is None or len(m_page.bill_items) == 0:
            if s_page is not None:
                merged.append(s_page)
            continue

        base_items: List[BillItem] = list(m_page.bill_items)
        seen_keys = set()

        def make_key(it: BillItem) -> Tuple[str, float, float, float]:
            return (
                (it.item_name or "").strip().lower(),
                round(float(it.item_amount or 0.0), 2),
                round(float(it.item_rate or 0.0), 2),
                round(float(it.item_quantity or 0.0), 3),
            )

        for it in base_items:
            seen_keys.add(make_key(it))

        if s_page is not None:
            for it in s_page.bill_items:
                k = make_key(it)
                if k not in seen_keys:
                    base_items.append(it)
                    seen_keys.add(k)

        # Decide page type
        page_type = (m_page.page_type or "").strip()
        if not page_type and s_page is not None:
            page_type = s_page.page_type

        page_no = m_page.page_no or (s_page.page_no if s_page else str(i + 1))

        merged.append(
            PageItems(
                page_no=str(page_no),
                page_type=page_type or "Bill Detail",
                bill_items=base_items,
            )
        )

    return merged


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

    default_rate = {
        k: (sum(v) / len(v)) for k, v in rates.items() if v
    }
    default_qty = {
        k: (sum(v) / len(v)) for k, v in qtys.items() if v
    }

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
#  Main Datathon Endpoint (POST) – HYBRID MAX-ACCURACY
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

    # === SCOUT STRATEGY & IMAGES ===
    scout_strategy = choose_scout_strategy(num_pages)
    scout_page_infos = build_page_infos(
        raw_pages,
        max_dim=scout_strategy["max_dim"],
        contrast=scout_strategy["contrast"],
        sharpness=scout_strategy["sharpness"],
        jpeg_quality=scout_strategy["jpeg_quality"],
        use_smart_crop=scout_strategy["use_smart_crop"],
    )

    # === MAVERICK STRATEGY & IMAGES (HIGH RES) ===
    maverick_strategy = choose_maverick_strategy(num_pages)
    mav_page_infos = build_page_infos(
        raw_pages,
        max_dim=maverick_strategy["max_dim"],
        contrast=maverick_strategy["contrast"],
        sharpness=maverick_strategy["sharpness"],
        jpeg_quality=maverick_strategy["jpeg_quality"],
        use_smart_crop=maverick_strategy["use_smart_crop"],
    )

    # 3. SCOUT BULK PASS
    scout_pages: List[PageItems] = []
    total_tokens = 0
    input_tokens = 0
    output_tokens = 0

    scout_batch_size = scout_strategy["batch_size"]
    for batch_start in range(0, num_pages, scout_batch_size):
        batch_end = min(batch_start + scout_batch_size, num_pages)
        batch_page_infos = scout_page_infos[batch_start:batch_end]

        raw_pages_batch, usage_batch = call_groq_for_batch(
            batch_page_infos,
            model_id=scout_strategy["model"],
            image_detail="auto",
        )

        total_tokens += usage_batch.total_tokens
        input_tokens += usage_batch.input_tokens
        output_tokens += usage_batch.output_tokens

        # Map batch-local page_no → global page_no
        for i, page_dict in enumerate(raw_pages_batch):
            global_page_no = batch_start + i + 1  # 1-based for entire document
            page_dict["page_no"] = str(global_page_no)
            page_items = reconcile_page_items(page_dict)
            scout_pages.append(page_items)

    if not scout_pages:
        elapsed = time.time() - start_time
        print(
            f"[BILL_EXTRACT] (Scout-only) pages={num_pages} items=0 total_amount=0.00 "
            f"tokens={total_tokens} time_sec={elapsed:.2f} (no items)"
        )
        return ExtractBillDataResponse(
            is_success=False,
            token_usage=TokenUsage(
                total_tokens=total_tokens,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
            message="Model did not return any page items (Scout).",
        )

    # 4. FULL MAVERICK PASS (HIGH ACCURACY)
    maverick_pages: List[PageItems] = []
    mav_batch_size = maverick_strategy["batch_size"]

    for batch_start in range(0, num_pages, mav_batch_size):
        batch_end = min(batch_start + mav_batch_size, num_pages)
        batch_page_infos = mav_page_infos[batch_start:batch_end]

        raw_pages_batch, usage_batch = call_groq_for_batch(
            batch_page_infos,
            model_id=maverick_strategy["model"],
            image_detail="high",
        )

        total_tokens += usage_batch.total_tokens
        input_tokens += usage_batch.input_tokens
        output_tokens += usage_batch.output_tokens

        for i, page_dict in enumerate(raw_pages_batch):
            global_page_no = batch_start + i + 1
            page_dict["page_no"] = str(global_page_no)
            page_items = reconcile_page_items(page_dict)
            maverick_pages.append(page_items)

    # 5. MERGE SCOUT + MAVERICK
    merged_pages = merge_scout_and_maverick_pages(
        scout_pages=scout_pages,
        mav_pages=maverick_pages,
    )

    # 6. SECOND PASS – fill numeric gaps using global patterns
    merged_pages = enrich_from_patterns(merged_pages)

    # 7. Aggregate
    data = aggregate_all_pages(merged_pages)

    token_usage = TokenUsage(
        total_tokens=total_tokens,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    # 8. Logging
    grand_total = compute_grand_total_amount(merged_pages)
    elapsed = time.time() - start_time

    print(
        f"[BILL_EXTRACT] HYBRID pages={num_pages} "
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
