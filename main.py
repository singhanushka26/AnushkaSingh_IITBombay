# main.py
import base64
import io
import json
import math
import mimetypes
import os
from typing import List, Optional, Tuple, Any, Dict

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

# Vision model (multimodal) – as you requested
GROQ_VISION_MODEL_ID = "meta-llama/llama-4-maverick-17b-128e-instruct"


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
    version="1.2.0",
    description=(
        "Extracts line items from bill / invoice documents using Groq vision models. "
        "Implements the exact response schema required by HackRx Datathon."
    ),
)


# ============================================================
#  Helpers – Download & Convert Documents
# ============================================================

def download_document(url: str) -> bytes:
    """
    Download the document from the given URL.

    Works with:
    - Direct image URLs (png, jpg, jpeg, webp, etc.)
    - Direct PDF URLs
    - Google Drive 'uc?export=download&id=...' style links
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

    We:
    - First try: mimetypes from the URL.
    - Then: detect PDFs via '%PDF' header.
    - Fallback: 'application/octet-stream'.
    """
    mime, _ = mimetypes.guess_type(url)
    if mime:
        return mime

    # crude PDF header check
    if content[:4] == b"%PDF":
        return "application/pdf"

    return "application/octet-stream"


def image_to_data_url(img: Image.Image, quality: int = 85) -> str:
    """
    Convert a PIL Image into a base64 JPEG data URL.

    Groq limits base64 images to ~4MB – we adaptively reduce JPEG quality.
    """
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


def document_to_page_images(url: str, content: bytes) -> List[str]:
    """
    Convert the downloaded document into a list of page images,
    each encoded as base64 data URLs.

    - If image: one page.
    - If PDF: one image per page.
    """
    mime = guess_mime_type(url, content)

    # Simple images
    if mime.startswith("image/"):
        try:
            img = Image.open(io.BytesIO(content)).convert("RGB")
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Unable to open image document: {e}",
            )
        return [image_to_data_url(img)]

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
        data_urls: List[str] = []
        for p in pages:
            img = p.convert("RGB")
            data_urls.append(image_to_data_url(img))
        return data_urls

    # Fallback: try opening as an image anyway
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
        return [image_to_data_url(img)]
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported document type: {mime}",
        )


# ============================================================
#  LLM Prompt – LONG, STRICT VERSION
# ============================================================

SYSTEM_PROMPT = """
You are an expert medical billing extraction engine.

Goal:
From a SINGLE PAGE of a bill/invoice image, you must extract ONLY the item-level rows
from the charges table(s). You must return STRICT JSON according to the given schema.

Definitions:
- A "line item" is one row in a charge table containing a description (test/procedure/room/etc.),
  with a quantity, a rate, and a total amount.
- Ignore patient demographics, addresses, headings like "HOSPITAL", "DETAIL FINAL BILL",
  and any non-item text.
- Many bills group items under section headers like "Radiological Investigation",
  "BED CHARGES", "CONSULTATION", "PATHOLOGY", "PHARMACY CHARGE".
  Section headers are NOT items.

VERY IMPORTANT RULES:
1. NEVER merge two distinct items into one JSON object.
   - Example: "RENAL FUNCTION TEST (RFT)" and "ELECTROLYTES" are two separate rows.

2. NEVER hallucinate nonexistent items.
   - If you are unsure, leave it out – do NOT invent.

3. Use the numeric columns exactly:
   - item_quantity = the value under "Qty" or "Qty/Hrs" or equivalent.
   - item_rate     = the "Rate" column (per unit).
   - item_amount   = the "Gross Amount", "Net Amt", "Total", or "Amt (Rs.)" column for that row.
   - If quantity is clearly missing but there is a total, assume quantity = 1 and rate = total.
   - If rate is missing but quantity and amount are visible, set rate = amount / quantity.
   - If a numeric value is BLANK / NOT VISIBLE, use 0.0 instead of null.

4. Use section totals ONLY as a sanity check.
   - Bills may show lines like "Total of PATHOLOGY : 10098.00" or "Grand Total : 73420.25".
   - Do NOT output these totals as items.
   - You CAN use them to check if the item_amount values are consistent,
     but do not include them in the JSON.

5. page_type classification:
   - "Bill Detail"  → detailed list of investigations/procedures/charges.
   - "Final Bill"   → summary-style, may still contain many line items plus grand total.
   - "Pharmacy"    → mostly medicines, drug names, quantity and rate.
   Choose the best label for THIS page only.

6. Every JSON object in bill_items must correspond to exactly ONE visible row
   of the charge table from this page.

7. If the document has multiple tables on the same page, include items from ALL tables
   (radiology, bed charges, consultation, pathology, pharmacy, etc.) – but still
   treat each physical row as a separate bill_items entry.

Output format:
Return STRICT JSON (no markdown, no comments) with exactly this shape:

{
  "page_no": "<page number as string>",
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
- Do NOT wrap the JSON in ``` marks.
- Do NOT add extra keys.
- Do NOT include subtotals, grand totals, taxes, discounts, or rounding lines as bill_items.
- All numeric fields must be valid numbers (no null, no empty string).
"""


def call_groq_for_page(page_no: int, image_data_url: str) -> Tuple[Dict[str, Any], TokenUsage]:
    """
    Call Groq Vision model via Responses API for a single page image.

    Returns:
        (parsed_json_dict, token_usage)
    """
    user_text = f"""
You are processing page number {page_no}.
Use page_no="{page_no}" in the JSON.
Classify page_type as one of: "Bill Detail", "Final Bill", "Pharmacy".
Remember: extract ONLY item-level rows from THIS PAGE.
"""

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
                    "content": [
                        {
                            "type": "input_text",
                            "text": user_text.strip(),
                        },
                        {
                            "type": "input_image",
                            "image_url": image_data_url,
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

    # Token usage from Groq Responses API
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

    # Text output
    raw_text = response.output_text.strip()

    # Parse JSON robustly
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        # Try to salvage JSON embedded in text
        first = raw_text.find("{")
        last = raw_text.rfind("}")
        if first != -1 and last != -1 and last > first:
            json_str = raw_text[first: last + 1]
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
#  Reconcile & Aggregate
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
        # remove commas in numbers like "52,868.25"
        s = s.replace(",", "")
        try:
            return float(s)
        except Exception:
            return 0.0
    # anything else → 0.0
    return 0.0


def clean_page_dict(page_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pre-clean the raw JSON dict from the model so that Pydantic parsing will not fail
    when numeric fields are null/empty/etc.
    """
    bill_items = page_dict.get("bill_items", [])
    cleaned_items = []
    for item in bill_items:
        if not isinstance(item, dict):
            continue
        cleaned_items.append(
            {
                "item_name": str(item.get("item_name", "")).strip(),
                "item_amount": _coerce_number(item.get("item_amount")),
                "item_rate": _coerce_number(item.get("item_rate")),
                "item_quantity": _coerce_number(item.get("item_quantity")),
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
            computed_rate = amount / qty
            item.item_rate = round(computed_rate, 4)
        elif amount and (not qty or qty == 0):
            item.item_quantity = 1.0
            item.item_rate = round(amount, 2)

    return page_items


def aggregate_all_pages(pages: List[PageItems]) -> ExtractBillDataResponseData:
    """
    Compute total_item_count. (Grand total is intentionally NOT included,
    to match the exact Datathon schema.)
    """
    total_items = sum(len(p.bill_items) for p in pages)
    return ExtractBillDataResponseData(
        pagewise_line_items=pages,
        total_item_count=total_items,
    )


# ============================================================
#  API Endpoint
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
    url_str = str(req.document)

    # 1. Download document
    content = download_document(url_str)

    # 2. Convert to per-page images (base64 data URLs)
    page_images = document_to_page_images(url_str, content)
    if not page_images:
        return ExtractBillDataResponse(
            is_success=False,
            message="No pages/images could be extracted from the document.",
        )

    all_pages: List[PageItems] = []
    total_tokens = 0
    input_tokens = 0
    output_tokens = 0

    # 3. Run Groq Vision model on each page
    for i, img_data_url in enumerate(page_images, start=1):
        parsed_json, usage = call_groq_for_page(page_no=i, image_data_url=img_data_url)
        page_items = reconcile_page_items(parsed_json)
        all_pages.append(page_items)

        total_tokens += usage.total_tokens
        input_tokens += usage.input_tokens
        output_tokens += usage.output_tokens

    # 4. Aggregate
    data = aggregate_all_pages(all_pages)

    token_usage = TokenUsage(
        total_tokens=total_tokens,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    return ExtractBillDataResponse(
        is_success=True,
        token_usage=token_usage,
        data=data,
    )
