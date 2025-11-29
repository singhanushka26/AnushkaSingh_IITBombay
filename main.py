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

# ============================================================
#  Groq Client (OpenAI-compatible)
# ============================================================

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

# Fast + good accuracy
GROQ_VISION_MODEL_ID = os.environ.get(
    "GROQ_VISION_MODEL_ID",
    "meta-llama/llama-4-vision-preview",
)

# Groq Vision limit: MAX 5 images per request
MAX_IMAGES_PER_REQUEST = 5


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
    version="7.0.0",
    description=(
        "Extracts line items from multi-page bill/invoice documents using "
        "Groq vision models with batching (max 5 images per request). "
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
    Slightly smaller than default for better speed on 30+ pages.
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
    Light contrast + sharpness boost to help faint numeric columns
    (like those IP CONSULTATION CHARGES pages).
    """
    img = ImageEnhance.Contrast(img).enhance(1.6)
    img = ImageEnhance.Sharpness(img).enhance(1.4)
    return img


def image_to_data_url(img: Image.Image, quality: int = 60) -> str:
    """
    Convert a PIL Image into a base64 JPEG data URL.

    - Resize to max_dim=950
    - Slight enhancement
    - JPEG quality ~60
    - Ensure < 4 MB per image
    """
    img = _resize_for_vision(img, max_dim=950)
    img = _enhance_for_vision(img)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b = buf.getvalue()

    while len(b) > 4 * 1024 * 1024 and quality > 30:
        quality -= 10
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        b = buf.getvalue()

    b64 = base64.b64encode(b).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


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
#  LLM Prompt – row-by-row, strict JSON
# ============================================================

SYSTEM_PROMPT = """
You are an expert medical BILL TABLE extraction engine.

RULES YOU MUST FOLLOW:

1. READ THE TABLE **ROW BY ROW**.
   - One visible table row = exactly one JSON object in bill_items.
   - Never merge or combine multiple rows into one JSON object.
   - Never split one row into multiple JSON objects.

2. SERVICE NAMES:
   - item_name must be copied as-is from the Service name column for that row.
   - Do not rephrase, summarize, or combine with other rows.
   - Do not group consumables (each consumable row is separate).

3. NUMBERS:
   - item_quantity = numeric value from the Qty column for that row.
   - item_rate     = numeric value from the Rate column.
   - item_amount   = numeric value from the Amount column.
   - If a numeric field is blank/unreadable, use 0.0 (do NOT hallucinate a new value).
   - Use plain numbers, no commas (e.g., 1797.56 not "1,797.56").

4. ROWS TO EXCLUDE:
   - DO NOT output totals, subtotals, net payable, discounts, taxes, or summary-only rows.
   - Examples to exclude: "TOTAL", "SUB TOTAL", "GRAND TOTAL", "NET AMOUNT PAYABLE",
     "AMOUNT RECEIVED", "BALANCE", "DISCOUNT", "ROUND OFF", "CGST", "SGST", "IGST", etc.

5. PAGE TYPE:
   - Choose one for each page: "Bill Detail", "Final Bill", or "Pharmacy".
   - Most tabular charge pages are "Bill Detail".
   - Pharmacy-dominated pages can be "Pharmacy".
   - Final summary pages are "Final Bill".

6. JSON ONLY:
   - Output must be a single JSON object.
   - No markdown, no explanations, no comments, no trailing commas.

JSON FORMAT (MANDATORY):

{
  "pagewise_line_items": [
    {
      "page_no": "1",
      "page_type": "Bill Detail" | "Final Bill" | "Pharmacy",
      "bill_items": [
        {
          "item_name": "string",
          "item_amount": float,
          "item_rate": float,
          "item_quantity": float
        }
      ]
    }
  ],
  "total_item_count": integer
}

- page_no is the 1-based index WITHIN THE CURRENT BATCH (as a string).
- bill_items may be [] for pages with no charge rows.
- total_item_count is the total number of bill_items across all pages in the batch.
"""


# ============================================================
#  JSON Repair Utility
# ============================================================

def try_json_load(text: str) -> dict:
    """
    Load JSON safely:
    - Strip markdown fences
    - Cut everything before first '{' and after last '}'
    - Fix common trailing comma issues
    """
    text = text.strip()

    # Remove fenced blocks if present
    if text.startswith("```"):
        # drop leading ```... and trailing ```
        text = text.strip("`")
        # after stripping backticks the braces logic still works

    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last == -1 or last <= first:
        raise ValueError("No valid JSON object found in model output.")

    trimmed = text[first:last + 1]

    # First attempt
    try:
        return json.loads(trimmed)
    except Exception:
        # Try simple trailing-comma fixes
        fixed = trimmed.replace(",}", "}").replace(",]", "]")
        return json.loads(fixed)


# ============================================================
#  Batched LLM Call
# ============================================================

def call_model_for_batch(
    batch_page_infos: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Sends up to MAX_IMAGES_PER_REQUEST images in a single LLM call.
    Returns:
      - list of page dicts ("page_no", "page_type", "bill_items")
      - dict with token usage
    """
    num_pages = len(batch_page_infos)

    header = f"""
You are given {num_pages} page image(s) from a hospital bill.

The images are in order:
- image 1 = page 1 in this batch
- image {num_pages} = page {num_pages} in this batch

For EACH batch page i (1-based):
- Use page_no = "<i>" as a string.
- Pick an appropriate page_type.
- Extract bill_items for that page only.
"""

    user_content: List[Dict[str, Any]] = [
        {"type": "input_text", "text": header.strip()}
    ]

    for idx, info in enumerate(batch_page_infos):
        batch_page_no = idx + 1
        user_content.append(
            {
                "type": "input_text",
                "text": f"PAGE {batch_page_no} IMAGE BELOW.",
            }
        )
        user_content.append(
            {
                "type": "input_image",
                "image_url": info["data_url"],
                "detail": "high",  # more accurate on small text
            }
        )

    try:
        response = client.responses.create(
            model=GROQ_VISION_MODEL_ID,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": SYSTEM_PROMPT.strip()}],
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
            detail=f"Groq API error: {e}",
        )

    usage = getattr(response, "usage", None)
    token_usage = {
        "total": int(getattr(usage, "total_tokens", 0) or 0),
        "input": int(getattr(usage, "input_tokens", 0) or 0),
        "output": int(getattr(usage, "output_tokens", 0) or 0),
    }

    raw_text = response.output_text.strip()

    # Parse JSON robustly
    try:
        parsed = try_json_load(raw_text)
    except Exception:
        # Log piece of raw for debugging
        print("[MODEL_RAW]", raw_text[:400])
        raise HTTPException(
            status_code=500,
            detail="Model response is not valid JSON.",
        )

    if not isinstance(parsed, dict) or "pagewise_line_items" not in parsed:
        raise HTTPException(
            status_code=500,
            detail="Model JSON does not contain 'pagewise_line_items'.",
        )

    raw_pages = parsed.get("pagewise_line_items", []) or []

    # Guarantee at least one page object per input page
    if len(raw_pages) < num_pages:
        for idx in range(len(raw_pages), num_pages):
            raw_pages.append(
                {
                    "page_no": str(idx + 1),
                    "page_type": "Bill Detail",
                    "bill_items": [],
                }
            )

    # If it returned extra pages, ignore extras
    raw_pages = raw_pages[:num_pages]

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

    Example:
        - Several 'IP CONSULTATION CHARGES' rows have amount=1000, qty=1.
        - Others have amount=0 (unreadable). We set:
              qty = 1, rate = 1000, amount = 1000.
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

            # Only try to "guess" when numbers are clearly missing / zero
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
    url_str = str(req.document)   # ensure plain string, not HttpUrl

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

    # 3. Process pages in batches of <= MAX_IMAGES_PER_REQUEST
    all_pages: List[PageItems] = []
    total_tokens = 0
    input_tokens = 0
    output_tokens = 0

    for batch_start in range(0, num_pages, MAX_IMAGES_PER_REQUEST):
        batch_end = min(batch_start + MAX_IMAGES_PER_REQUEST, num_pages)
        batch_page_infos = page_infos[batch_start:batch_end]

        raw_pages_batch, usage_batch = call_model_for_batch(batch_page_infos)

        total_tokens += usage_batch["total"]
        input_tokens += usage_batch["input"]
        output_tokens += usage_batch["output"]

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

    # 4. Enrich missing numeric fields using global patterns
    all_pages = enrich_from_patterns(all_pages)

    # 5. Aggregate
    data = aggregate_all_pages(all_pages)

    token_usage = TokenUsage(
        total_tokens=total_tokens,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    # 6. Logging
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
