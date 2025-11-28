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
from PIL import Image, ImageStat
from openai import OpenAI

# ============================================================
#  Groq Client (OpenAI-compatible)
# ============================================================

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

# Vision model (multimodal)
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
    version="2.0.0",
    description=(
        "Extracts line items from bill / invoice documents using Groq vision models "
        "plus OCR assistance. Implements the exact response schema required by "
        "HackRx Datathon."
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


def image_to_data_url(img: Image.Image, quality: int = 85) -> str:
    """
    Convert a PIL Image into a base64 JPEG data URL.

    Groq limits base64 images to ~4MB – we adaptively reduce JPEG quality.
    """
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b = buf.getvalue()

    # Adaptive compression
    while len(b) > 4 * 1024 * 1024 and quality > 30:
        quality -= 10
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        b = buf.getvalue()

    b64 = base64.b64encode(b).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# ---------------- OCR + Blank-page detection ----------------

def extract_ocr_text(img: Image.Image) -> str:
    """
    Run Tesseract OCR on the given image.

    If pytesseract is not installed or Tesseract is missing,
    we fail silently and return an empty string (vision model still works).
    """
    try:
        import pytesseract
    except Exception:
        # Optional dependency – no hard failure
        return ""

    try:
        # You can tweak config if needed
        text = pytesseract.image_to_string(img)
        return text or ""
    except Exception:
        return ""


def is_mostly_blank(img: Image.Image, ocr_text: str) -> bool:
    """
    Heuristic: detect near-blank pages so we don't send
    blank scans / separators to the LLM.

    Criteria:
    - Very bright (mean grayscale > 245)
    - AND OCR text length < 30 characters
    """
    gray = img.convert("L")
    stat = ImageStat.Stat(gray)
    mean = stat.mean[0] if stat.mean else 255.0

    if mean > 245 and len(ocr_text.strip()) < 30:
        return True
    return False


def document_to_page_infos(url: str, content: bytes) -> List[Dict[str, Any]]:
    """
    Convert the downloaded document into a list of PAGE INFOS:

        {
          "pil_image": <PIL.Image>,
          "data_url": "data:image/jpeg;base64,...",
          "ocr_text": "...",
          "is_blank": bool
        }

    - If image: one page.
    - If PDF: one image per page.
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

        ocr_text = extract_ocr_text(img)
        info = {
            "pil_image": img,
            "data_url": image_to_data_url(img),
            "ocr_text": ocr_text,
            "is_blank": is_mostly_blank(img, ocr_text),
        }
        page_infos.append(info)
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

        for p in pages:
            img = p.convert("RGB")
            ocr_text = extract_ocr_text(img)
            info = {
                "pil_image": img,
                "data_url": image_to_data_url(img),
                "ocr_text": ocr_text,
                "is_blank": is_mostly_blank(img, ocr_text),
            }
            page_infos.append(info)
        return page_infos

    # Fallback: try as image
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported document type: {mime}",
        )

    ocr_text = extract_ocr_text(img)
    info = {
        "pil_image": img,
        "data_url": image_to_data_url(img),
        "ocr_text": ocr_text,
        "is_blank": is_mostly_blank(img, ocr_text),
    }
    page_infos.append(info)
    return page_infos


# ============================================================
#  LLM Prompt – OCR-aware, strict JSON
# ============================================================

SYSTEM_PROMPT = """
You are an expert medical billing extraction engine.

Goal:
From a SINGLE PAGE IMAGE of a bill/invoice, extract ONLY the item-level rows
from the charges table(s). Return STRICT JSON according to the given schema.

Definitions:
- A "line item" is one row in a charge table containing a description
  (test, procedure, medicine, room, bed, etc.) with a quantity, a rate,
  and a total amount for that row.
- Ignore patient demographics, addresses, logos, headers like "HOSPITAL",
  "MEDICOS", "DETAIL FINAL BILL", and any non-item text.
- Many bills group items under section headers like "Radiological Investigation",
  "BED CHARGES", "CONSULTATION", "PATHOLOGY", "PHARMACY CHARGE".
  Section headers are NOT items.

VERY IMPORTANT RULES:
1. Process ONLY THIS PAGE IMAGE. Do NOT invent rows from other pages.
   Every JSON row must correspond to a visible row within THIS page image.

2. NEVER merge two distinct rows into one JSON object.
   Example: "RENAL FUNCTION TEST (RFT)" and "ELECTROLYTES" are two separate rows.

3. NEVER hallucinate nonexistent items.
   If you are unsure about a row, skip it instead of inventing.

4. Use the numeric columns exactly:
   - item_quantity = the value under "Qty", "QTY", "Qty/Hrs", etc.
   - item_rate     = the "Rate", "RATE" or per-unit price column.
   - item_amount   = the "Amount", "Net Amt", "Total", "Amt (Rs.)" column.
   - If quantity is clearly missing but there is a total, assume quantity = 1 and rate = total.
   - If rate is missing but quantity and amount are visible, set rate = amount / quantity.
   - If a numeric value is BLANK / NOT VISIBLE, use 0.0 instead of null.

5. Section totals and grand totals:
   - Rows like "Total of PATHOLOGY", "Total of PHARMACY", "Grand Total",
     "Net Amount Payable", "SUB TOTAL", "CGST", "SGST", "TAX ON", "ROUND OFF"
     are NOT items. Do NOT include them in bill_items.
   - You may use them only mentally to sanity check your amounts.

6. page_type classification:
   - "Bill Detail"  → detailed list of investigations/procedures/charges.
   - "Final Bill"   → summary-style, may still contain many line items plus grand total.
   - "Pharmacy"    → mostly medicines, drug names, quantity and rate.
   Choose the best label for THIS page only.

7. Multiple tables:
   - If the page shows multiple tables (radiology, bed charges, pharmacy, etc.),
     include items from ALL tables, but still treat each physical row
     as a separate bill_items entry.

8. Numeric types:
   - All numeric fields must be valid JSON numbers (no null, no empty strings).
   - Use period as decimal separator. Do NOT use commas inside numbers.

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
"""


def call_groq_for_page(
    page_no: int,
    image_data_url: str,
    ocr_text: str,
) -> Tuple[Dict[str, Any], TokenUsage]:
    """
    Call Groq Vision model for a single page image, with OCR text as hint.
    """
    # Truncate OCR text to keep prompt reasonable
    ocr_text_snippet = (ocr_text or "").strip()
    if len(ocr_text_snippet) > 2000:
        ocr_text_snippet = ocr_text_snippet[:2000]

    user_text = f"""
You are processing page number {page_no} of a multi-page bill.

Use page_no="{page_no}" in the JSON.
Classify page_type as one of: "Bill Detail", "Final Bill", "Pharmacy".

Below is OCR text extracted from THIS PAGE ONLY.
Use it as a hint to read small fonts and numbers, but always match
it to the visual table rows in the image.

OCR_TEXT_START
{ocr_text_snippet}
OCR_TEXT_END

Remember:
- Extract ONLY item-level rows that are PHYSICALLY PRESENT in THIS PAGE IMAGE.
- Do NOT carry over rows from previous or next pages.
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

    # Robust JSON parsing
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
    Pre-clean the raw JSON dict from the model so that Pydantic parsing will not fail
    when numeric fields are null/empty/etc.
    """
    bill_items = page_dict.get("bill_items", []) or []
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


def dedupe_all_pages(pages: List[PageItems]) -> List[PageItems]:
    """
    De-duplicate clearly repeated items across pages.

    We treat items with the same:
        (page_type, normalized_name, rate, qty, amount)
    as duplicates and keep only the first occurrence.

    This is mainly to avoid hallucinated repeated segments
    (e.g., same pharmacy list repeated on multiple pages).
    """
    seen: set = set()
    deduped_pages: List[PageItems] = []

    for p in pages:
        new_items: List[BillItem] = []
        for item in p.bill_items:
            key = (
                p.page_type,
                item.item_name.strip().lower(),
                round(float(item.item_rate), 2),
                round(float(item.item_quantity), 2),
                round(float(item.item_amount), 2),
            )
            if key in seen:
                # duplicate – likely hallucinated repetition
                continue
            seen.add(key)
            new_items.append(item)

        p.bill_items = new_items
        deduped_pages.append(p)

    return deduped_pages


def aggregate_all_pages(pages: List[PageItems]) -> ExtractBillDataResponseData:
    """
    Compute total_item_count. Grand total is intentionally NOT included,
    to match the exact Datathon schema.
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

    # 2. Convert to per-page infos (image + data_url + OCR)
    page_infos = document_to_page_infos(url_str, content)
    if not page_infos:
        return ExtractBillDataResponse(
            is_success=False,
            message="No pages/images could be extracted from the document.",
        )

    all_pages: List[PageItems] = []
    total_tokens = 0
    input_tokens = 0
    output_tokens = 0

    # 3. Run Groq Vision model on each NON-BLANK page
    logical_page_no = 1
    for info in page_infos:
        if info.get("is_blank"):
            # skip obviously blank pages
            continue

        data_url = info["data_url"]
        ocr_text = info.get("ocr_text", "")

        parsed_json, usage = call_groq_for_page(
            page_no=logical_page_no,
            image_data_url=data_url,
            ocr_text=ocr_text,
        )

        page_items = reconcile_page_items(parsed_json)
        # Ensure page_no is consistent with our logical numbering
        page_items.page_no = str(logical_page_no)
        all_pages.append(page_items)

        total_tokens += usage.total_tokens
        input_tokens += usage.input_tokens
        output_tokens += usage.output_tokens

        logical_page_no += 1

    if not all_pages:
        return ExtractBillDataResponse(
            is_success=False,
            message="All pages were detected as blank; nothing to extract.",
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

    return ExtractBillDataResponse(
        is_success=True,
        token_usage=token_usage,
        data=data,
    )
