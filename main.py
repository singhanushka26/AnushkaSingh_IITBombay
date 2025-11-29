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

# Speed-optimized default: Scout (you can override via env)
GROQ_VISION_MODEL_ID = os.environ.get(
    "GROQ_VISION_MODEL_ID",
    "meta-llama/llama-4-scout-17b-16e-instruct",
)


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
    version="4.0.0",
    description=(
        "Extracts line items from multi-page bill / invoice documents using "
        "Groq vision models. Single-shot multi-page call for high accuracy "
        "and low latency, with strict HackRx Datathon response schema."
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
    Uses a smaller max_dim to handle up to ~30 pages quickly.
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

    We aggressively compress for speed + token limit:
    - Resize to max_dim=1100
    - JPEG quality ~60 (enough for text to be readable for the model)
    """
    img = _resize_for_vision(img, max_dim=1100)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b = buf.getvalue()

    # Safety: ensure < 4 MB per image as per Groq base64 limits.
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
          "page_index": int,
          "data_url": "data:image/jpeg;base64,..."
        }

    - If image: one page.
    - If PDF: one image per page.
    OCR is deliberately NOT used here to save CPU and time.
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
#  LLM Prompt – Multi-page, strict JSON, no double-count
# ============================================================

SYSTEM_PROMPT = """
You are an expert medical BILL ITEM extraction engine for multi-page hospital bills.

You must follow these goals:

1) Capture EVERY genuine line item on ALL pages (high recall).
2) Do NOT double-count or duplicate the same line item (no over-count).
3) Make the sum of all `item_amount` values across all pages as close as possible
   to the FINAL TOTAL printed in the bill.
4) Respect the exact JSON format required by the HackRx Datathon.

---------------- DOCUMENT & TERMINOLOGY ----------------

You will receive multiple page images of a hospital bill, in order:
    Page 1, Page 2, ..., Page N.

Each page may contain:
- Headers (logo, hospital name, patient details, dates).
- One or more TABLES of charges:
    * Investigations / Lab tests
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
- A "LINE ITEM" is a single row of a table representing a single charge
  with a meaningful description and a numeric amount.

For each LINE ITEM, you must output:
    * item_name:   description of the charge
    * item_quantity: quantity / days / units
    * item_rate:   rate per unit
    * item_amount: net amount for that item AFTER discounts, BEFORE tax
      (the number printed in that row for that item)

Section titles and totals:
- Titles like "PATHOLOGY", "RADIOLOGY", "PHARMACY CHARGES", "BED CHARGES",
  "CONSULTATION", etc. are NOT line items.
- Rows like "Total PATHOLOGY", "LAB TOTAL", "PHARMACY TOTAL",
  "SUB TOTAL", "TOTAL", "GRAND TOTAL", "NET AMOUNT PAYABLE",
  "ROUND OFF", "DISCOUNT", "GST", "CGST", "SGST", "IGST" are NOT items.
You MUST NOT output these rows as bill_items.

---------------- CRITICAL RULES ----------------

1. PROCESS ALL PAGES TOGETHER
   - You see all page images in order.
   - Use the full context to avoid double-counting repeated summary pages.
   - Only output items that correspond to REAL rows in the tables.

2. DO NOT MISS ITEMS
   - For every visible row with a description AND an amount, output one bill_items entry.
   - If a row is obviously a header/title or a total/summary, skip it.

3. DO NOT DOUBLE-COUNT
   - If an item is listed as a detailed row on page 2 and again summarized on
     the "Final Bill" page, you must ONLY output it ONCE.
   - If the same item appears multiple times as separate rows (for different days
     or quantities), then output multiple entries, but only WHEN there really
     are separate rows.
   - Do NOT emit multiple identical entries for a single visual row.

4. NUMERIC FIELDS
   - item_quantity: read from "Qty", "QTY", "No. of Days", etc.
   - item_rate:     from "Rate", "RATE", "Charges per day", etc.
   - item_amount:   from "Amount", "Net Amt", "Total", etc.
   Rules:
   - If quantity is missing but an amount is present:
         quantity = 1.0
         rate = amount
   - If rate is missing but quantity and amount are present:
         rate = amount / quantity (rounded sensibly).
   - If any numeric field is blank/unreadable:
         use 0.0 (not null / not empty string).
   - Always use numeric JSON values (e.g. 1200.5, not "1,200.5").

5. PAGE TYPE
   For each page, set:
       page_type = "Bill Detail" | "Final Bill" | "Pharmacy"

   - "Bill Detail" → general pages with mixed charges tables.
   - "Pharmacy"    → pages dominated by medicine/drug items.
   - "Final Bill"  → summary pages with high-level totals and sometimes
                     compressed line items.

   Choose the most appropriate label for each page.

6. OUTPUT STRUCTURE (GLOBAL, FOR ALL PAGES)

You MUST output ONE SINGLE JSON OBJECT of the form:

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
- page_no is the 1-based page index as a STRING.
- You MUST include one object in pagewise_line_items for EVERY input page,
  in order, from 1 to N.
- bill_items can be an empty list [] when a page has no line items.
- total_item_count is the TOTAL number of items across ALL bill_items on ALL pages.

STRICT REQUIREMENTS:
- Do NOT wrap the JSON in ``` or any markdown.
- Do NOT add any keys other than pagewise_line_items and total_item_count
  at the top level.
- Do NOT add extra keys inside page objects or item objects.
- Do NOT output totals, sub-totals, tax rows, or duplicate summary lines
  as bill_items.
"""


def call_groq_for_document(
    page_infos: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], TokenUsage]:
    """
    Single Groq Vision call for the entire multi-page document.
    Passes all pages as images in order, with a strong global system prompt.
    """
    num_pages = len(page_infos)

    # Short user instruction describing pages
    user_text = f"""
You are given a hospital bill with {num_pages} page image(s).

The images will be provided in order: first image is page 1, second is page 2,
and so on up to page {num_pages}.

For EACH page i (1-based), you must:
- Set page_no = "{'{'}i{'}'}"
- Choose page_type from: "Bill Detail", "Final Bill", "Pharmacy"
- Extract bill_items for that page ONLY.
Then, you must output ONE combined JSON object for all pages, matching the
exact format described in the system instructions.
"""

    # Build content list: one user message with text + all images
    user_content: List[Dict[str, Any]] = [
        {
            "type": "input_text",
            "text": user_text.strip(),
        }
    ]

    for idx, info in enumerate(page_infos):
        page_no = idx + 1
        user_content.append(
            {
                "type": "input_text",
                "text": f"PAGE {page_no} IMAGE BELOW.",
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

    # Robust JSON parsing (strip ```json fences etc.)
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

    return parsed, token_usage


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

    We treat items with identical:
        (page_type, normalized_name, rate, qty, amount)
    as duplicates and keep only the first occurrence.

    Protects against double-counting when summary pages re-list items
    that are already fully detailed elsewhere.
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
    if not page_infos:
        elapsed = time.time() - start_time
        print(f"[BILL_EXTRACT] pages=0 items=0 total_amount=0.00 tokens=0 time_sec={elapsed:.2f} (no pages)")
        return ExtractBillDataResponse(
            is_success=False,
            message="No pages/images could be extracted from the document.",
        )

    # 3. One Groq Vision call for ALL pages
    raw_parsed, usage = call_groq_for_document(page_infos)

    # 4. Normalize shape: ensure we have "pagewise_line_items"
    if isinstance(raw_parsed, dict) and "pagewise_line_items" in raw_parsed:
        raw_pages = raw_parsed.get("pagewise_line_items", []) or []
    elif isinstance(raw_parsed, list):
        # Fallback: model returned list of pages directly
        raw_pages = raw_parsed
    else:
        raise HTTPException(
            status_code=500,
            detail="Model JSON does not contain 'pagewise_line_items' list.",
        )

    # Ensure one page object per input page (even if empty)
    if len(raw_pages) < len(page_infos):
        # Pad missing pages with empty structures
        for idx in range(len(raw_pages), len(page_infos)):
            raw_pages.append(
                {
                    "page_no": str(idx + 1),
                    "page_type": "Bill Detail",
                    "bill_items": [],
                }
            )

    all_pages: List[PageItems] = []
    for idx, page_dict in enumerate(raw_pages):
        # Force page_no to match 1-based order
        page_dict["page_no"] = str(idx + 1)
        page_items = reconcile_page_items(page_dict)
        all_pages.append(page_items)

    if not all_pages:
        elapsed = time.time() - start_time
        print(f"[BILL_EXTRACT] pages={len(page_infos)} items=0 total_amount=0.00 "
              f"tokens={usage.total_tokens} time_sec={elapsed:.2f} (no items)")
        return ExtractBillDataResponse(
            is_success=False,
            message="Model did not return any page items.",
        )

    # 5. De-duplicate obviously repeated items across pages
    all_pages = dedupe_all_pages(all_pages)

    # 6. Aggregate
    data = aggregate_all_pages(all_pages)
    token_usage = usage

    # 7. Compute a simple "grand total amount" for logging
    grand_total = compute_grand_total_amount(all_pages)
    elapsed = time.time() - start_time

    # Railway log line – helps you debug performance & quality
    print(
        f"[BILL_EXTRACT] pages={len(page_infos)} "
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
