# main.py
import base64
import io
import json
import math
import mimetypes
import os
import time
from typing import List, Optional, Tuple, Any, Dict, Set

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from pdf2image import convert_from_bytes
from PIL import Image
from openai import OpenAI

# ============================================================
#  Groq Client (OpenAI-compatible)
# ============================================================

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

# Balanced mode: fast + good accuracy
GROQ_VISION_MODEL_ID = os.environ.get(
    "GROQ_VISION_MODEL_ID",
    "meta-llama/llama-4-scout-17b-16e-instruct",
)

# Groq Vision limit: MAX 5 images per request
MAX_IMAGES_PER_REQUEST = 5


# ============================================================
#  Pydantic Schemas – EXACTLY as per Datathon spec
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
    version="5.0.0",
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
    """
    Download the document from the given URL.
    Supports:
    - Direct image URLs (png, jpg, jpeg, webp, etc.)
    - Direct PDF URLs
    - Public file links that resolve to the above
    """
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
    """
    Guess mime type based on URL extension or magic bytes.
    """
    mime, _ = mimetypes.guess_type(url)
    if mime:
        return mime

    if content[:4] == b"%PDF":
        return "application/pdf"

    return "application/octet-stream"


def _resize_for_vision(img: Image.Image, max_dim: int = 1100) -> Image.Image:
    """
    Downscale to reduce tokens & latency while keeping table details readable.
    Smaller max_dim → faster + cheaper, still fine for Scout.
    """
    w, h = img.size
    scale = max(w, h) / float(max_dim)
    if scale <= 1.0:
        return img
    new_w = int(w / scale)
    new_h = int(h / scale)
    return img.resize((new_w, new_h), Image.LANCZOS)


def image_to_data_url(img: Image.Image, quality: int = 60) -> str:
    """
    Convert a PIL Image into a base64 JPEG data URL.

    - Resize to max_dim=1100
    - JPEG quality ~60 (enough for text to be readable for the model)
    - Ensure < 4 MB per image as per Groq base64 limits.
    """
    img = _resize_for_vision(img, max_dim=1100)

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

    - If image: one page.
    - If PDF: one image per page.
    OCR is NOT used for speed.
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
#  LLM Prompt – Multi-page (batched), strict JSON, no double-count
# ============================================================

SYSTEM_PROMPT = """
You are an expert medical BILL ITEM extraction engine for hospital bills.

Your goals:

1) Capture EVERY genuine line item from the tables (high recall).
2) Do NOT double-count or duplicate the same line item (no over-count).
3) Make the sum of all `item_amount` values across all pages as close as possible
   to the FINAL TOTAL printed in the bill.
4) Respect the exact JSON structure required by the HackRx Datathon.

---------------- DOCUMENT & TERMINOLOGY ----------------

You will receive a *subset* of pages from a hospital bill:
    Page 1, Page 2, ..., Page K (within this batch).

Each page may contain:
- Headers (hospital name, patient details, dates).
- One or more TABLES of charges:
    * Lab tests / Investigations
    * Radiology / Imaging
    * Pharmacy / Medicines
    * Bed / Room / ICU charges
    * Operation theatre / Procedure charges
    * Consultation / Doctor visit charges
    * Nursing / Maintenance / Service charges
- Sub-total rows per section.
- Final summary rows such as:
    "Total", "Sub Total", "Grand Total", "Net Amount Payable",
    "Amount Received", "Balance", "Discount", "GST", "CGST", "SGST", etc.

Definitions:
- A "LINE ITEM" is a single concrete charge row with a meaningful description
  and a numeric amount.

For each LINE ITEM, you must output:
    * item_name:    description of the charge
    * item_quantity: quantity / days / units
    * item_rate:    rate per unit
    * item_amount:  net amount for that item AFTER discounts, BEFORE tax
      (exact number printed in that row for that item when possible)

Section titles and totals:
- Titles like "PATHOLOGY", "RADIOLOGY", "PHARMACY CHARGES", "BED CHARGES",
  "CONSULTATION", etc. are NOT line items.
- Rows like "Total PATHOLOGY", "LAB TOTAL", "PHARMACY TOTAL",
  "SUB TOTAL", "TOTAL", "GRAND TOTAL", "NET AMOUNT PAYABLE",
  "ROUND OFF", "DISCOUNT", "GST", "CGST", "SGST", "IGST" are NOT items.
You MUST NOT output these rows as bill_items.

---------------- CRITICAL RULES ----------------

1. PROCESS ONLY THE PAGES IN THIS BATCH
   - You see K page images in order for this batch.
   - For each page i in this batch, you must output data for THAT page only.

2. DO NOT MISS ITEMS
   - For every visible row with description + amount in a table, output one bill_items entry.
   - Skip pure headers/titles and pure total/summary rows.

3. DO NOT DOUBLE-COUNT
   - If an item is shown as a detailed line item and then repeated as a section total
     or final total, ONLY output the detailed line, not the total.
   - Do NOT create two JSON entries for a single visual table row.

4. NUMERIC FIELDS
   - item_quantity: from "Qty", "QTY", "No. of Days", etc.
   - item_rate:     from "Rate", "RATE", "Charges per day", etc.
   - item_amount:   from "Amount", "Net Amt", "Total", etc.
   Rules:
   - If quantity is missing but amount is present:
         quantity = 1.0
         rate = amount
   - If rate is missing but quantity and amount are present:
         rate = amount / quantity (rounded sensibly).
   - If any numeric field is blank/unreadable:
         use 0.0 (not null / empty string).
   - Use numeric JSON values only (e.g. 1200.5, not "1,200.5").

5. PAGE TYPE
   For each page in this batch, set:
       page_type = "Bill Detail" | "Final Bill" | "Pharmacy"

   - "Bill Detail" → general pages with mixed charges tables.
   - "Pharmacy"    → pages dominated by medicines/drugs.
   - "Final Bill"  → summary pages with totals and sometimes compressed lines.

Choose the most appropriate label per page.

---------------- OUTPUT STRUCTURE (PER BATCH) ----------------

You MUST output ONE SINGLE JSON OBJECT:

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
    },
    {
      "page_no": "2",
      "page_type": "Bill Detail" | "Final Bill" | "Pharmacy",
      "bill_items": [ ... ]
    }
  ],
  "total_item_count": <integer>
}

Where:
- page_no is the 1-based index WITHIN THIS BATCH (as a STRING).
- There must be exactly one object in pagewise_line_items for EACH page in this batch.
- bill_items can be [] if a page has no charge lines.
- total_item_count is the TOTAL number of items across ALL bill_items
  on ALL pages in this batch.

STRICT REQUIREMENTS:
- Do NOT wrap the JSON in ``` or any markdown.
- Do NOT add any keys other than pagewise_line_items and total_item_count
  at the top level.
- Do NOT add extra keys inside page objects or item objects.
- Do NOT output totals, sub-totals, tax rows, or duplicate summary lines
  as bill_items.
"""


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

Then return ONE combined JSON object for this batch, matching exactly the
structure described in the system instructions.
"""

    # Build content list: one user message with text + all images in this batch
    user_content: List[Dict[str, Any]] = [
        {
            "type": "input_text",
            "text": user_text.strip(),
        }
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
            model=GROQ_VISION_MODEL_ID,
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": SYSTEM_PROMPT.strip(),
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
            detail=f"Groq API error: {e}",
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

    # Robust JSON parsing (strip ```json fences etc. if present)
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        first = raw_text.find("{")
        last = raw_text.rfind("}")
        if first != -1 and last != -1 and last > first:
            json_str = raw_text[first:last + 1]
            try:
                parsed = json.loads(json_str)
            except Exception:
                raise HTTPException(
                    status_code=500,
                    detail=f"Model response is not valid JSON: {raw_text[:200]}",
                )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Model response is not valid JSON: {raw_text[:200]}",
            )

    # Normalise: extract list of pages from batch JSON
    if isinstance(parsed, dict) and "pagewise_line_items" in parsed:
        raw_pages = parsed.get("pagewise_line_items", []) or []
    elif isinstance(parsed, list):
        raw_pages = parsed
    else:
        raise HTTPException(
            status_code=500,
            detail="Model JSON for batch does not contain 'pagewise_line_items' list.",
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

    # If model returned extra pages, ignore extras
    raw_pages = raw_pages[:num_batch_pages]

    return raw_pages, token_usage


# ============================================================
#  Reconcile, Clean & Aggregate
# ============================================================

def _coerce_number(x: Any) -> float:
    """
    Coerce a potentially messy numeric value (None, "", "—", etc.) to float.
    """
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
    Also performs within-page exact duplicate removal (same name + numbers).
    """
    bill_items = page_dict.get("bill_items", []) or []
    cleaned_items: List[Dict[str, Any]] = []

    seen_keys: Set[Tuple[str, float, float, float]] = set()

    for item in bill_items:
        if not isinstance(item, dict):
            continue

        name = str(item.get("item_name", "")).strip()
        amount = _coerce_number(item.get("item_amount"))
        rate = _coerce_number(item.get("item_rate"))
        qty = _coerce_number(item.get("item_quantity"))

        key = (
            name.lower(),
            round(amount, 2),
            round(rate, 4),
            round(qty, 4),
        )
        if key in seen_keys:
            # Drop exact duplicates on the same page
            continue
        seen_keys.add(key)

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
                # Trust arithmetic over noisy OCR when mismatch is large
                item.item_amount = round(computed, 2)
        elif amount and qty and qty != 0:
            computed_rate = amount / qty
            item.item_rate = round(computed_rate, 4)
        elif amount and (not qty or qty == 0):
            item.item_quantity = 1.0
            item.item_rate = round(amount, 2)

    return page_items


def dedupe_all_pages(pages: List[PageItems]) -> List[PageItems]:
    """
    De-duplicate clearly repeated items across pages.

    Items with identical:
        (page_type, normalized_name, rate, qty, amount)
    are treated as duplicates; we keep only the first occurrence.

    This protects against double-counting when summary pages re-list items
    already fully detailed elsewhere.
    """
    seen: Set[Tuple[str, str, float, float, float]] = set()
    deduped_pages: List[PageItems] = []

    for p in pages:
        new_items: List[BillItem] = []
        for item in p.bill_items:
            key = (
                p.page_type.strip().lower(),
                item.item_name.strip().lower(),
                round(float(item.item_rate), 4),
                round(float(item.item_quantity), 4),
                round(float(item.item_amount), 2),
            )
            if key in seen:
                # duplicate – likely repeated summary row
                continue
            seen.add(key)
            new_items.append(item)

        p.bill_items = new_items
        deduped_pages.append(p)

    return deduped_pages


def aggregate_all_pages(pages: List[PageItems]) -> ExtractBillDataResponseData:
    """
    Compute total_item_count. Grand total is intentionally NOT included in
    the schema (organizers will compute it from bill_items).
    """
    total_items = sum(len(p.bill_items) for p in pages)
    return ExtractBillDataResponseData(
        pagewise_line_items=pages,
        total_item_count=total_items,
    )


def compute_grand_total_amount(pages: List[PageItems]) -> float:
    """
    Utility: sum of all item_amounts – for logging only.
    """
    total = 0.0
    for p in pages:
        for item in p.bill_items:
            total += float(item.item_amount)
    return round(total, 2)


# ============================================================
#  Health Check (GET) – handles evaluator GET pings
# ============================================================

@app.get("/extract-bill-data")
def health_check():
    """
    Simple GET endpoint so health checks don't see 405.
    The evaluators will use POST for actual scoring.
    """
    return {
        "message": "Health OK. Use POST /extract-bill-data with JSON body "
                   '{"document": "<public image/PDF URL>"} to extract bill data.'
    }


# ============================================================
#  Main Datathon Endpoint (POST)
# ============================================================

@app.post("/extract-bill-data", response_model=ExtractBillDataResponse)
def extract_bill_data(req: ExtractBillDataRequest):
    """
    Main Datathon endpoint.

    Request:
        {
          "document": "<public URL to image or PDF>"
        }

    Response (SUCCESS, 200):
        {
          "is_success": true,
          "token_usage": {
            "total_tokens": int,
            "input_tokens": int,
            "output_tokens": int
          },
          "data": {
            "pagewise_line_items": [...],
            "total_item_count": int
          }
        }
    """
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

    # 3. Process pages in batches of <= MAX_IMAGES_PER_REQUEST
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

    # 4. De-duplicate obviously repeated items across pages
    all_pages = dedupe_all_pages(all_pages)

    # 5. Aggregate
    data = aggregate_all_pages(all_pages)

    token_usage = TokenUsage(
        total_tokens=total_tokens,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    # 6. Compute a simple "grand total amount" for logging
    grand_total = compute_grand_total_amount(all_pages)
    elapsed = time.time() - start_time

    # Railway log line – helps you debug performance & quality
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
