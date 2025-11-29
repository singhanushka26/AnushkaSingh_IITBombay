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

# Vision model (multimodal, good for production)
GROQ_VISION_MODEL_ID = "meta-llama/llama-4-maverick-17b-128e-instruct"
# If you want to test Scout instead, change to:
# GROQ_VISION_MODEL_ID = "meta-llama/llama-4-scout-17b-16e-instruct"

# High-recall mode: number of crops per page
NUM_CROPS_PER_PAGE = 3  # top / middle / bottom slices


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
    title="Bajaj Datathon Bill Extraction API – High Recall",
    version="3.0.0",
    description=(
        "High-recall extraction of line items from bill / invoice documents "
        "using Groq LLaMA-4 Vision with Tesseract OCR assistance. "
        "Multi-crop per page, repeated row-preserving, Datathon schema compliant."
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


def image_to_data_url(img: Image.Image, quality: int = 85) -> str:
    """
    Convert a PIL Image into a base64 JPEG data URL.

    Groq limits base64 images to ~4MB – adaptively reduce JPEG quality.
    """
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b = buf.getvalue()

    # Adaptive compression loop
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

    If pytesseract or Tesseract is not available, returns "" silently.
    """
    try:
        import pytesseract
    except Exception:
        return ""

    try:
        text = pytesseract.image_to_string(img)
        return text or ""
    except Exception:
        return ""


def is_mostly_blank(img: Image.Image, ocr_text: str) -> bool:
    """
    For high-recall mode we disable blank skipping by always returning False.

    If you ever want blank detection back, use something like:
        gray = img.convert("L")
        stat = ImageStat.Stat(gray)
        mean = stat.mean[0] if stat.mean else 255.0
        return bool(mean > 245 and len(ocr_text.strip()) < 30)
    """
    return False


def document_to_page_infos(url: str, content: bytes) -> List[Dict[str, Any]]:
    """
    Convert the downloaded document into a list of PAGE INFOS:

        {
          "pil_image": <PIL.Image>,
          "data_url": "data:image/jpeg;base64,...",  # full page
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
        page_infos.append(
            {
                "pil_image": img,
                "data_url": image_to_data_url(img),
                "ocr_text": ocr_text,
                "is_blank": is_mostly_blank(img, ocr_text),
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

        for p in pages:
            img = p.convert("RGB")
            ocr_text = extract_ocr_text(img)
            page_infos.append(
                {
                    "pil_image": img,
                    "data_url": image_to_data_url(img),
                    "ocr_text": ocr_text,
                    "is_blank": is_mostly_blank(img, ocr_text),
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

    ocr_text = extract_ocr_text(img)
    page_infos.append(
        {
            "pil_image": img,
            "data_url": image_to_data_url(img),
            "ocr_text": ocr_text,
            "is_blank": is_mostly_blank(img, ocr_text),
        }
    )
    return page_infos


# ============================================================
#  LLM Prompt – OCR-aware, strong & repetition-safe
# ============================================================

SYSTEM_PROMPT = """
You are an ADVANCED MEDICAL BILL EXTRACTION ENGINE designed for high-volume 
hospital invoices, repeated consultation logs, pharmacy sheets, lab reports, 
and multi-page detailed bills. Your job is to EXTRACT EVERY SINGLE PHYSICAL 
LINE-ITEM ROW printed on THIS PAGE IMAGE (or cropped region of this page).

CRITICAL BEHAVIOR:
============================================================
(1) DO NOT group or merge multiple rows into one.
(2) DO NOT deduplicate.
(3) DO NOT infer items from previous/next pages.
(4) DO NOT hallucinate rows or columns.
(5) IF TWO ROWS LOOK IDENTICAL, extract BOTH as separate entries.
(6) If a row spans two visual lines but is one logical row,
    MERGE those lines and output a single JSON item.
(7) If two items are visually separate rows, NEVER merge them.

ABOUT THESE PAGES:
============================================================
• Bills may contain 10–70+ rows on one page.
• Many rows can be repetitive (e.g., “IP CONSULTATION CHARGES” repeated 40 times).
• Some pages may contain multiple tables: investigations, pharmacy, consultation.
• Some pages include blank or zero-valued columns (e.g., "0.00") which must be preserved.

WHAT COUNTS AS A LINE ITEM:
============================================================
A “line item” is a PHYSICAL TABLE ROW that has:
• A description (test name, charge name, drug name, ward charge, procedure name, etc.)
• A quantity (Qty, QTY, Qty/Hrs, No. of units)
• A rate (Rate, Price, Unit Price)
• A total / amount (Amount, Total, Net Amt, Line Total)

Each PHYSICAL ROW → EXACTLY ONE JSON OBJECT in bill_items.

WHAT TO IGNORE:
============================================================
• Page headers, footers, serial numbers, dates, patient info
• “Name”, “IP No”, “Bill No”, “UHID”, “Admn No”, etc.
• Section headers (e.g., “CONSULTATION”, “INVESTIGATION CHARGES”, “OTHERS”)
• Summary rows or financial totals such as:
  “Total”, “Sub Total”, “Net Amount Payable”, “Grand Total”,
  “ROUND OFF”, “CGST / SGST / IGST”, “DISCOUNT”, “Tax on Item”, etc.

NUMERIC EXTRACTION:
============================================================
• item_quantity: exact numeric quantity from Qty/QTY/QTY/Hrs column.
• item_rate: exact per-unit rate.
• item_amount: the total for that row.
• If quantity is clearly missing but a single total exists, assume quantity = 1 and rate = total.
• If rate is missing but quantity and amount are visible, set rate = amount / quantity.
• If a numeric field is blank or unreadable, use 0.0 instead of null.
• If amount is very close to rate × quantity, keep the printed amount; do NOT over-correct.

PAGE TYPE CLASSIFICATION:
============================================================
Choose ONE page_type based ONLY on THIS PAGE (or crop of this page):
• "Bill Detail"  – investigations, procedures, ward/room charges, consultations, etc.
• "Pharmacy"     – medicines, drugs, syrups, injections.
• "Final Bill"   – summary-style bill with a smaller list plus grand totals.

MULTI-TABLE PAGES:
============================================================
If the page (or crop) contains multiple tables:
• Extract items from ALL tables you see in this image region.
• Process each visible row independently.
• Preserve natural top-to-bottom order as much as possible.

ANTI-HALLUCINATION RULES:
============================================================
• NEVER add rows that are not visibly present.
• NEVER invent section names, descriptions, or numeric values.
• NEVER copy rows from OCR text if they do not match an actual row in the image.
• When in doubt, SKIP a row instead of hallucinating.

OUTPUT FORMAT:
============================================================
Return STRICT JSON ONLY (no markdown, no commentary), with this exact shape:

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

Do NOT wrap JSON in ``` marks.
Do NOT add extra keys.
"""


def call_groq_for_region(
    page_no: int,
    image_data_url: str,
    ocr_text: str,
) -> Tuple[Dict[str, Any], TokenUsage]:
    """
    Call Groq Vision model for a single page REGION (crop) image,
    with OCR text from the full page as a hint.
    """
    # Truncate OCR text to keep prompt size reasonable
    ocr_text_snippet = (ocr_text or "").strip()
    if len(ocr_text_snippet) > 2000:
        ocr_text_snippet = ocr_text_snippet[:2000]

    user_text = f"""
You are processing PAGE {page_no} of a multi-page bill.

This image is a REGION (crop) from that page. Extract ONLY the line items 
that are VISIBLY PRESENT inside this region.

Use page_no="{page_no}" in the JSON.

Below is OCR text extracted from the FULL PAGE. Use it only as a noisy hint
to help read small fonts and numbers. DO NOT invent items that are not visible
in this region.

OCR_TEXT_START
{ocr_text_snippet}
OCR_TEXT_END

Remember:
- Each PHYSICAL ROW in this region → exactly one JSON object.
- Repeated rows must be output MULTIPLE times.
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
    Pre-clean the raw JSON dict from the model so that Pydantic parsing will not
    fail when numeric fields are null/empty/etc.
    """
    bill_items = page_dict.get("bill_items", []) or []
    cleaned_items: List[Dict[str, Any]] = []

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
            # Only adjust when mismatch is very large and amount looks wrong
            if math.isfinite(computed) and abs(computed - amount) > 5 * EPS:
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
    Compute total_item_count. Grand total is intentionally NOT included,
    to match the exact Datathon schema.

    NOTE: We DO NOT deduplicate here, to preserve repeated rows like
    multiple IP CONSULTATION CHARGES entries.
    """
    total_items = sum(len(p.bill_items) for p in pages)
    return ExtractBillDataResponseData(
        pagewise_line_items=pages,
        total_item_count=total_items,
    )


# ============================================================
#  API Endpoint – High Recall Multi-Crop
# ============================================================

@app.post("/extract-bill-data", response_model=ExtractBillDataResponse)
def extract_bill_data(req: ExtractBillDataRequest):
    """
    Main Datathon endpoint – High Recall Mode.

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

    # 3. Run Groq Vision model on each NON-BLANK page with multi-crop
    logical_page_no = 1
    for info in page_infos:
        if info.get("is_blank"):
            # For safety we keep the branch; currently is_blank is always False.
            continue

        pil_page: Image.Image = info["pil_image"]
        full_ocr_text = info.get("ocr_text", "")

        # ----- Generate crops -----
        width, height = pil_page.size
        crop_h = height // NUM_CROPS_PER_PAGE or height
        crops: List[Image.Image] = []
        for i in range(NUM_CROPS_PER_PAGE):
            top = i * crop_h
            bottom = height if i == NUM_CROPS_PER_PAGE - 1 else (i + 1) * crop_h
            crops.append(pil_page.crop((0, top, width, bottom)))

        # ----- Call model for each crop -----
        crop_results: List[Dict[str, Any]] = []
        crop_page_types: List[str] = []

        for crop in crops:
            crop_data_url = image_to_data_url(crop)
            parsed_json, usage = call_groq_for_region(
                page_no=logical_page_no,
                image_data_url=crop_data_url,
                ocr_text=full_ocr_text,
            )

            # accumulate usage
            total_tokens += usage.total_tokens
            input_tokens += usage.input_tokens
            output_tokens += usage.output_tokens

            # store results
            crop_results.append(parsed_json)
            pt = str(parsed_json.get("page_type", "") or "").strip()
            if pt:
                crop_page_types.append(pt)

        # ----- Merge crops into one logical page -----
        merged_items_raw: List[Dict[str, Any]] = []
        for cr in crop_results:
            items = cr.get("bill_items", []) or []
            if isinstance(items, list):
                merged_items_raw.extend(items)

        # pick majority page_type if possible
        if crop_page_types:
            from collections import Counter
            pt_counts = Counter(crop_page_types)
            page_type = pt_counts.most_common(1)[0][0]
        else:
            page_type = "Bill Detail"

        raw_page_dict: Dict[str, Any] = {
            "page_no": str(logical_page_no),
            "page_type": page_type,
            "bill_items": merged_items_raw,
        }

        page_items = reconcile_page_items(raw_page_dict)
        # Ensure page_no exactly matches our logical counter
        page_items.page_no = str(logical_page_no)
        all_pages.append(page_items)

        logical_page_no += 1

    if not all_pages:
        return ExtractBillDataResponse(
            is_success=False,
            message="All pages were detected as blank; nothing to extract.",
        )

    # 4. Aggregate without deduplication (preserve repeated rows)
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
