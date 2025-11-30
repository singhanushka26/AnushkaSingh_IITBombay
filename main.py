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

# Optional OCR – used for row density heuristics
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

# Fast model (used only as fallback in this EXTREME version)
GROQ_VISION_MODEL_SCOUT = os.environ.get(
    "GROQ_VISION_MODEL_SCOUT",
    "meta-llama/llama-4-scout-17b-16e-instruct",
)

# Primary high-accuracy model (Maverick everywhere)
GROQ_VISION_MODEL_MAVERICK = os.environ.get(
    "GROQ_VISION_MODEL_MAVERICK",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
)

# Groq Vision limit: MAX 5 images per request
MAX_IMAGES_PER_REQUEST = 5

# Soft global time budget (not enforced strictly)
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
    version="10.0.0-a-plus-extreme",
    description=(
        "A+ EXTREME: Maverick-heavy, high-resolution, OCR-aware, "
        "with single-page refinement for suspicious pages. "
        "Designed for maximum accuracy under a soft ~120s budget."
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
    Conservative heuristic crop to remove headers/footers and side margins.
    Slightly less aggressive than older variant for accuracy.
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
    quality: int = 65,
    max_bytes: int = 4 * 1024 * 1024,
) -> bytes:
    """Encode image to JPEG with given quality, shrinking if too large."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b = buf.getvalue()
    while len(b) > max_bytes and quality > 40:
        quality -= 5
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
#  Dynamic Strategy Selection – A+ EXTREME
# ============================================================

def choose_strategy(num_pages: int) -> Dict[str, Any]:
    """
    A+ EXTREME strategy (accuracy-first):

    - ALWAYS use Maverick for bulk.
    - High resolution + higher JPEG quality.
    - OCR row estimation on most pages.
    - Maverick refinement on suspicious pages.

    This can be heavy on tokens/time, but aims at maximum accuracy.
    """
    per_page_budget = GLOBAL_TIME_BUDGET_SEC / max(1, num_pages)

    strategy: Dict[str, Any] = {
        "bulk_model": GROQ_VISION_MODEL_MAVERICK,
        "bulk_batch_size": 1,
        "bulk_max_dim": 1100,
        "contrast": 1.5,
        "sharpness": 1.4,
        "jpeg_quality": 65,
        "use_smart_crop": True,
        "use_ocr": True,
        "ocr_max_pages": 30,
        "use_refine": True,
        "refine_model": GROQ_VISION_MODEL_MAVERICK,
        "refine_limit": num_pages,  # tuned below
    }

    if num_pages <= 6:
        # Very small docs – go crazy on quality
        strategy.update(
            {
                "bulk_batch_size": 1,
                "bulk_max_dim": 1150,
                "jpeg_quality": 70,
                "ocr_max_pages": num_pages,
                "refine_limit": num_pages,  # can refine all if needed
            }
        )
    elif num_pages <= 12:
        # Small/medium docs
        strategy.update(
            {
                "bulk_batch_size": 2,
                "bulk_max_dim": 1100,
                "jpeg_quality": 65,
                "ocr_max_pages": min(20, num_pages),
                "refine_limit": min(num_pages, 8),
            }
        )
    elif num_pages <= 20:
        # Medium/large docs
        strategy.update(
            {
                "bulk_batch_size": 2 if per_page_budget > 5 else 3,
                "bulk_max_dim": 1050,
                "jpeg_quality": 60,
                "ocr_max_pages": min(25, num_pages),
                "refine_limit": min(num_pages, 10),
            }
        )
    else:
        # Very large docs – still Maverick, but slightly tighter
        strategy.update(
            {
                "bulk_batch_size": 3,
                "bulk_max_dim": 1000,
                "jpeg_quality": 58,
                "ocr_max_pages": min(30, num_pages),
                "refine_limit": min(num_pages, 12),
            }
        )

    # Always respect batch limit
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
    ocr_max_pages = strategy.get("ocr_max_pages", 0)

    for idx, img in enumerate(raw_pages):
        img_proc = img.convert("RGB")
        if strategy.get("use_smart_crop", True):
            img_proc = smart_crop(img_proc)
        img_proc = enhance_image(
            img_proc,
            contrast_factor=strategy.get("contrast", 1.5),
            sharpness_factor=strategy.get("sharpness", 1.4),
        )
        img_proc = resize_image_max_dim(
            img_proc, strategy.get("bulk_max_dim", 1100)
        )

        # OCR row estimation for heuristics
        ocr_rows = 0
        if (
            strategy.get("use_ocr", True)
            and OCR_AVAILABLE
            and idx < ocr_max_pages
        ):
            ocr_rows = estimate_table_rows(img_proc)

        jpeg_bytes = image_to_jpeg_bytes(
            img_proc,
            quality=strategy.get("jpeg_quality", 65),
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
#  LLM Prompts – BULK + REFINEMENT
# ============================================================

SYSTEM_PROMPT_BULK = """
You are an expert hospital BILL ITEM extraction engine.

You receive a BATCH of page images from a single hospital bill.

For EACH page image:
- Read all charge tables.
- For EVERY visible row that represents a real charge (description + amount),
  output ONE entry in bill_items for that page.
- DO NOT output:
  - totals (TOTAL, SUBTOTAL, GRAND TOTAL, NET AMOUNT PAYABLE, BALANCE, etc.)
  - discounts, concessions, round-off rows
  - taxes (GST, IGST, SGST, CGST, etc.)
  - headings, section titles, or empty rows.

Numeric rules:
- item_quantity: from columns like Qty, No. of days, Units, etc.
- item_rate:     from Rate, Charges per day, Per unit, etc.
- item_amount:   from Amount, Net Amount, etc.
- If quantity missing but amount visible: quantity = 1.0, rate = amount.
- If rate missing but quantity & amount visible: rate = amount / quantity.
- If a numeric field is unreadable: 0.0.

Repeated rows:
- If the same row appears MANY TIMES visually (same description and numbers),
  you must output one bill_items entry per visual row.
  DO NOT collapse multiple rows into a single entry.

Output MUST be a single JSON object with this exact schema:

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

Definitions:
- page_no is the 1-based index within THIS BATCH, as a STRING ("1", "2", ...).
- page_type:
    - "Bill Detail" for detailed charge pages,
    - "Final Bill" for summary/settlement pages,
    - "Pharmacy" for drug/medicine item pages.
- total_item_count = total number of bill_items across all pages in this batch.

STRICT RULES:
- Return JSON ONLY. No markdown, no explanations, no comments.
- Do NOT add extra top-level keys.
- Do NOT include totals / taxes / summary-only lines as bill_items.
"""

SYSTEM_PROMPT_REFINEMENT = """
You are a precise hospital BILL ITEM extraction engine.

You will see ONLY ONE page image of a hospital bill.

Task for this single page:
- Read all charge tables.
- For EVERY visible row that represents a real charge (description + amount),
  output ONE entry in bill_items.
- DO NOT output:
  - totals (TOTAL, SUBTOTAL, GRAND TOTAL, NET AMOUNT PAYABLE, BALANCE, etc.)
  - discounts, concessions, round-off rows
  - taxes (GST, IGST, SGST, CGST, etc.)
  - headings, section titles, or empty rows.
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

Numeric rules:
- item_quantity: from Qty / No. of days / Units, etc.
- item_rate: from Rate / Per day / Per unit, etc.
- item_amount: from Amount / Net Amount, etc.
- If quantity missing but amount visible: quantity = 1.0, rate = amount.
- If rate missing but quantity & amount visible: rate = amount / quantity.
- If a numeric field is unreadable: 0.0.

STRICT:
- JSON only (no markdown, no comments, no extra keys).
"""

# ============================================================
#  Response Text Helper (Groq Responses)
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
    - Replaces common non-JSON tokens (NaN, Infinity, -Infinity → 0).
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

    # Take outermost {...}
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        text = text[first : last + 1]

    # Light sanitization
    sanitized = text
    sanitized = sanitized.replace("NaN", "0")
    sanitized = sanitized.replace("Infinity", "0")
    sanitized = sanitized.replace("-Infinity", "0")

    sanitized = sanitized.replace(",]", "]")
    sanitized = sanitized.replace(", ]", "]")
    sanitized = sanitized.replace(",}", "}")
    sanitized = sanitized.replace(", }", "}")

    try:
        return json.loads(sanitized)
    except json.JSONDecodeError:
        # Final attempt with tight outer braces
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
#  LLM Call Helpers – BULK + REFINEMENT
# ============================================================

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
                "detail": "high",  # high detail for Maverick
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
        # Fallback to Scout in case of 429 / temporary Maverick failure
        fallback_msg = str(e)
        try:
            response = client.responses.create(
                model=GROQ_VISION_MODEL_SCOUT,
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
        except Exception as e2:
            raise HTTPException(
                status_code=503,
                detail=f"Groq API error (bulk batch, Maverick+Scout fallback): {fallback_msg} | {e2}",
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
    Single-page refinement using Maverick (or Scout fallback).
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
                        {
                            "type": "input_text",
                            "text": SYSTEM_PROMPT_REFINEMENT.strip(),
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
        # Fallback to Scout once
        fallback_msg = str(e)
        try:
            response = client.responses.create(
                model=GROQ_VISION_MODEL_SCOUT,
                input=[
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "input_text",
                                "text": SYSTEM_PROMPT_REFINEMENT.strip(),
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": user_content,
                    },
                ],
            )
        except Exception as e2:
            raise HTTPException(
                status_code=503,
                detail=f"Groq API error (refine single-page, Maverick+Scout fallback): {fallback_msg} | {e2}",
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
#  Main Datathon Endpoint (POST) – A+ EXTREME
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

    # 3. Choose A+ EXTREME strategy & build page_infos
    strategy = choose_strategy(num_pages)
    page_infos = build_page_infos(raw_pages, strategy)

    # 4. Bulk pass – Maverick on all pages (with Scout fallback inside)
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

    # 5. Maverick refinement on suspicious pages
    if strategy.get("use_refine", True) and strategy.get("refine_limit", 0) > 0:
        suspicious_indices: List[int] = []
        for idx, p in enumerate(all_pages):
            ocr_rows = page_infos[idx].get("ocr_rows", 0)
            n_items = len(p.bill_items)
            page_type = (p.page_type or "").strip().lower()

            # Heuristics:
            very_sparse = (ocr_rows >= 10 and n_items <= max(3, ocr_rows // 3))
            almost_empty = (n_items <= 1 and ocr_rows >= 6)
            suspicious_final = (page_type == "final bill" and n_items < 10)

            if very_sparse or almost_empty or suspicious_final:
                suspicious_indices.append(idx)

        refine_limit = strategy["refine_limit"]
        suspicious_indices = suspicious_indices[:refine_limit]

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

    # 6. SECOND PASS – fill numeric gaps using global patterns
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
