# main.py
import base64
import io
import json
import math
import mimetypes
import os
from typing import List, Optional, Tuple, Any, Dict, Set

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
# For maximum accuracy we keep Maverick as default.
GROQ_VISION_MODEL_ID = os.environ.get(
    "GROQ_VISION_MODEL_ID",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
)

# Hard accuracy mode (single full-page call, strong prompt)
HARD_ACCURACY_MODE = True  # keep True for Datathon


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
    version="3.0.0",
    description=(
        "Extracts line items from bill / invoice documents using Groq vision models "
        "with Tesseract OCR assistance. Implements the exact response schema "
        "required by HackRx Datathon and is tuned to minimize missed items and "
        "double counting while matching the printed bill totals."
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


def _resize_for_vision(img: Image.Image, max_dim: int = 1600) -> Image.Image:
    """
    Light downscale to reduce tokens & latency while keeping details.
    Keeps aspect ratio. Only downsizes if larger than max_dim.
    """
    w, h = img.size
    scale = max(w, h) / float(max_dim)
    if scale <= 1.0:
        return img
    new_w = int(w / scale)
    new_h = int(h / scale)
    return img.resize((new_w, new_h), Image.LANCZOS)


def image_to_data_url(img: Image.Image, quality: int = 80) -> str:
    """
    Convert a PIL Image into a base64 JPEG data URL.

    Groq limits base64 images to ~4MB – adaptively reduce JPEG quality.
    """
    # Optional resize for speed/token control
    img = _resize_for_vision(img, max_dim=1600)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b = buf.getvalue()

    # Adaptive compression loop
    while len(b) > 4 * 1024 * 1024 and quality > 40:
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
    Very conservative blank-page detector:
    - High brightness AND
    - very low variance AND
    - almost no OCR text

    To avoid missing any bill content, this function errs on the side
    of treating pages as NON-blank.
    """
    # If there is any non-trivial OCR text, treat as non-blank
    if len((ocr_text or "").strip()) > 50:
        return False

    gray = img.convert("L")
    stat = ImageStat.Stat(gray)
    mean = stat.mean[0] if stat.mean else 255.0
    var = stat.var[0] if stat.var else 0.0

    # Very bright, very low-variance, and almost no text
    if mean > 248 and var < 5.0 and len((ocr_text or "").strip()) < 10:
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
        page_infos.append(
            {
                "pil_image": img,
                "data_url": image_to_data_url(img),
                "ocr_text": ocr_text,
                # VERY conservative: we do not treat pages as blank lightly
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
#  LLM Prompt – OCR-aware, high-precision + no double-count
# ============================================================

SYSTEM_PROMPT = """
You are an expert medical BILL ITEM extraction engine designed for an evaluation
where BOTH of the following matter:

1) Every genuine line item on the bill must be captured (high recall).
2) There must be NO double counting or duplication of the same line item.

The evaluator will:
- Compare the sum of all your extracted `item_amount` values against the
  FINAL TOTAL shown on the bill.
- Check that your line items match the descriptions and numeric values printed
  in the bill tables.
Your goal is to make the AI-extracted total as close as possible to the actual
bill final total WITHOUT missing items or double-counting.

---------- DOCUMENT STRUCTURE & TERMINOLOGY ----------

You are given a SINGLE PAGE IMAGE of a multi-page bill/invoice.

Typical structure:
- Hospital or clinic header (logo, name, address).
- Patient information (name, age, ID, dates).
- One or more TABLES containing charges:
    * Investigations/Pathology/Lab tests
    * Radiology / Imaging
    * Procedures / Operation charges
    * Consultation / Doctor visit charges
    * Bed / Room / Ward charges
    * Nursing, Maintenance, Service charges
    * Pharmacy / Medicines
- Optional sections with headings like:
    "PATHOLOGY", "RADIOLOGY", "PHARMACY CHARGES", "BED CHARGES", etc.
- Sub-total rows for each section.
- Final summaries: "Total", "Grand Total", "Net Amount Payable",
  "Amount Received", "Balance", "Discount", "GST", "CGST", "SGST", etc.

Definitions (VERY IMPORTANT):
- A "LINE ITEM" is one logical row describing a single charge, such as:
    - A CONCRETE test/procedure: "RENAL FUNCTION TEST (RFT)", "ELECTROLYTES"
    - A bed charge: "Room Ward Charges", "ICU Bed Charges"
    - A consultation: "IP CONSULTATION CHARGES (Dr. ...)"
    - A pharmacy item: a medicine or drug name with qty, rate, and amount.
- Each line item should have:
    * item_name: textual description for that charge.
    * item_quantity: how many units / days / hours, etc.
    * item_rate: per-unit price.
    * item_amount: net amount for that line item AFTER discounts, BEFORE tax
      (use the number printed in the row for that item).

Section headers and totals:
- Section titles like "PATHOLOGY", "RADIOLOGY", "PHARMACY CHARGES",
  "BED CHARGES", "CONSULTATION", etc. are NOT items.
- Summary/total rows like "Total PATHOLOGY", "LAB TOTAL", "PHARMACY TOTAL",
  "SUB TOTAL", "TOTAL", "GRAND TOTAL", "NET AMOUNT PAYABLE", "ROUND OFF",
  "GST", "CGST", "SGST", "IGST", "DISCOUNT", "TAX ON" are NOT items.

You MUST NOT output these rows as bill_items.
Instead, you implicitly use them as a sanity check: the sum of your item_amount
values for all rows on all pages should match these totals as closely as possible,
but you do not output the total rows themselves.

---------- CRITICAL RULES: NO MISSING, NO DOUBLE COUNTING ----------

1. PROCESS ONLY THIS PAGE IMAGE.
   - Do NOT invent items from other pages.
   - Every JSON object must correspond to a visible row on THIS page.

2. DO NOT MISS items:
   - For every visible row that clearly has a description + numeric amount,
     you must extract one bill_items entry.
   - If a row has no visible amount or is clearly not a charge (e.g., header),
     skip it.

3. DO NOT DOUBLE COUNT:
   - If the same line item is repeated multiple times with the same description,
     same quantity, same rate, and same amount (i.e., visually the same row only once),
     output it ONLY ONCE.
   - If the bill clearly lists the same item multiple times on separate rows
     (e.g., same test done on different days), then you may output multiple entries
     BUT only if they are really separate rows in the table.
   - Never output multiple identical JSON objects that correspond to a SINGLE visual row.

4. HANDLING QUANTITY / RATE / AMOUNT:
   - item_quantity = numeric value under "Qty", "QTY", "QTY/Hrs", "No. of days", etc.
   - item_rate     = numeric value under "Rate", "RATE", "Per day", "Charge", etc.
   - item_amount   = numeric value under "Amount", "Amt (Rs.)", "Net Amt", "Total", etc.
   - If quantity is clearly missing but there is a total amount for the row:
       -> assume quantity = 1.0 and rate = amount.
   - If rate is missing but quantity and amount are visible:
       -> rate = amount / quantity (rounded sensibly).
   - If a numeric field is BLANK / unreadable:
       -> set it to 0.0 (do NOT use null or empty string).

5. HANDLING SUB-TOTALS AND FINAL TOTAL:
   - Sub-total rows (per section) and the final total row are NOT individual items.
   - You may mentally cross-check that "sum of item_amount" for that section/page
     roughly matches the sub-total shown, but you MUST NOT add them as bill_items.
   - The final evaluation will compute:
       AI_Total = sum of all item_amount across all bill_items in the entire document
     and compare it to the printed final total on the bill.
   - Your responsibility is to:
       - include every genuine item row
       - NOT double-count the same row
       - ensure the numbers on each row are consistent with the bill.

6. page_type classification:
   - "Bill Detail"  → typical detailed list of line items (most pages).
   - "Pharmacy"    → mainly medicines / drugs with Qty, Rate, Amount.
   - "Final Bill"  → summary-style page that may repeat line items or show
                     only a compressed version plus final totals.
   Choose the best label FOR THIS PAGE ONLY.

7. MULTIPLE TABLES ON THE SAME PAGE:
   - If the page has multiple tables of charges (e.g., investigations + bed
     charges + consultation), extract items from ALL of them.
   - Still obey the rule: never treat section headers or totals as items.

8. NUMERIC TYPES:
   - All numeric fields must be valid JSON numbers (no null, no empty strings).
   - Use "." as decimal separator.
   - Do NOT put commas inside numbers (e.g., use 1200.5 not "1,200.5").

---------- OUTPUT FORMAT (STRICT JSON) ----------

You MUST output STRICT JSON (no markdown, no backticks) with this exact shape:

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
- Do NOT add extra keys at any level.
- Do NOT output totals, sub-totals, or tax rows as bill_items.
- Do NOT output the same (name, quantity, rate, amount) row multiple times
  unless they clearly correspond to separate visible rows in the table.
"""


def call_groq_for_page(
    page_no: int,
    image_data_url: str,
    ocr_text: str,
) -> Tuple[Dict[str, Any], TokenUsage]:
    """
    Call Groq Vision model for a single page image, with OCR text as hint.
    """

    # Truncate OCR text to keep prompt size reasonable & fast
    ocr_text_snippet = (ocr_text or "").strip()
    if len(ocr_text_snippet) > 1200:  # shorter than before → faster, cheaper
        ocr_text_snippet = ocr_text_snippet[:1200]

    user_text = f"""
You are processing page number {page_no} of a multi-page medical bill.

Use page_no="{page_no}" in the JSON.
Classify page_type as exactly one of:
- "Bill Detail"
- "Final Bill"
- "Pharmacy"

Below is OCR text extracted from THIS PAGE ONLY (may contain noise).
Use it to help read small fonts and numbers, but always confirm by looking
at the table structure in the image.

OCR_TEXT_START
{ocr_text_snippet}
OCR_TEXT_END

Remember:
- Extract ONLY item-level rows that are PHYSICALLY PRESENT in THIS PAGE IMAGE.
- Do NOT bring items from other pages.
- Do NOT miss any genuine line items.
- Do NOT double-count the same line item.
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


def clean_page_dict(page_dict: Dict[str, Any]) -> Dict[str, Any]]:
    """
    Pre-clean the raw JSON dict from the model so that Pydantic parsing will not
    fail when numeric fields are null/empty/etc.
    Also performs within-page exact duplicate removal at dictionary level.
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

        # Exact quadruple key for dedupe (within this page)
        key = (
            name.lower(),
            round(amount, 2),
            round(rate, 4),
            round(qty, 4),
        )
        if key in seen_keys:
            # LLM hallucinated a duplicate of the same row → drop it
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
                # Trust arithmetic over noisy OCR when there is a big mismatch
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

    This mainly protects against accidental double extraction of the
    SAME scan / SAME bill, not against legitimate multiple charges.
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
                # duplicate – likely hallucination or repeated extraction
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

    # 3. Run Groq Vision model on each NON-blank page
    logical_page_no = 1
    for info in page_infos:
        # VERY conservative: almost everything is treated as non-blank
        if info.get("is_blank"):
            # Skip only pages that are clearly blank
            logical_page_no += 1
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
