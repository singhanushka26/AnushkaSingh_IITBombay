# main.py
import base64
import io
import json
import math
import mimetypes
import os
import re
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

GROQ_VISION_MODEL_ID = os.environ.get(
    "GROQ_VISION_MODEL_ID",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
)

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
    version="4.0.0",
    description=(
        "Hybrid Vision + OCR pipeline for extracting line items from medical bills. "
        "Strictly follows HackRx Datathon response schema and focuses on high recall "
        "with minimal double-counting."
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


def _resize_for_vision(img: Image.Image, max_dim: int = 1600) -> Image.Image:
    """Downscale large images for speed + token control."""
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
    Groq limits base64 images to ~4MB – adaptively reduce quality.
    """
    img = _resize_for_vision(img, max_dim=1600)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b = buf.getvalue()

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
    Hybrid approach: if pytesseract is missing, gracefully fall back to "".
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
    Conservative blank-page detector:
    - If there is any decent OCR text → NOT blank
    - Otherwise, use very high brightness + low variance threshold
    """
    if len((ocr_text or "").strip()) > 50:
        return False

    gray = img.convert("L")
    stat = ImageStat.Stat(gray)
    mean = stat.mean[0] if stat.mean else 255.0
    var = stat.var[0] if stat.var else 0.0

    if mean > 248 and var < 5.0 and len((ocr_text or "").strip()) < 10:
        return True
    return False


def document_to_page_infos(url: str, content: bytes) -> List[Dict[str, Any]]:
    """
    Convert the downloaded document into list of PAGE INFOS:

        {
          "pil_image": <PIL.Image>,
          "data_url": "data:image/jpeg;base64,...",
          "ocr_text": "...",
          "is_blank": bool
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
                    "Ensure poppler is installed in the environment. "
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

    # Fallback: try as image
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
#  LLM Prompt – high recall + no double count
# ============================================================

SYSTEM_PROMPT = """
You are an expert medical BILL ITEM extraction engine.

Goals:
1) Capture EVERY genuine line item (high recall).
2) Avoid any double-counting of the same line item.

The evaluator will:
- Sum all `item_amount` across all pages.
- Compare that to the printed FINAL TOTAL on the bill.
- Check that each extracted line matches the bill tables.

You get ONE PAGE IMAGE at a time.
You MUST only use content that is visibly present on THIS PAGE.
Do NOT invent items from other pages.

A VALID line item row has:
- A description of the service/test/medicine/charge.
- A numeric amount (and usually quantity and rate).

Examples:
- Bed/room charges
- ICU / ward charges
- Consultation/doctor charges
- Pathology / radiology tests
- Operation / procedure charges
- IPD / OT / Nursing / Service charges
- Pharmacy items (medicines, consumables, devices)

NOT items (must be ignored):
- Section headers: "PATHOLOGY", "RADIOLOGY", "PHARMACY CHARGES",
  "BED CHARGES", "IPD CONSUMABLE CHARGES" if used as a heading only, etc.
- Page headers with hospital / patient info.
- Summary rows: "SUB TOTAL", "SUBTOTAL", "TOTAL", "GRAND TOTAL",
  "NET AMOUNT", "NET AMOUNT PAYABLE", "TOTAL AMOUNT",
  "DISCOUNT", "ROUND OFF", "GST", "CGST", "SGST", "IGST", "TAX".

These summaries can be used mentally to sanity-check totals but must
NOT appear as bill_items.

NUMERIC FIELDS:
- item_quantity → Qty / No. of days / units (default 1.0 if missing)
- item_rate     → Rate / Unit price (if missing but quantity & amount known,
                   infer rate = amount / quantity)
- item_amount   → Net amount of that row, AFTER discount and BEFORE taxes.

If any numeric field is unreadable, set it to 0.0 (never null/empty).

page_type:
- "Bill Detail" → detailed line items (most pages)
- "Pharmacy"   → mainly medicines / consumables with qty, rate, amount
- "Final Bill" → summary-style page with compressed items + final totals

You MUST choose exactly one page_type for each page.

Output STRICT JSON, no markdown, no explanations.
"""


# ========== Robust JSON Parsing / Repair ==========

def _strip_code_fences(text: str) -> str:
    """Remove ```json ... ``` fences if model adds them."""
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        # remove first fence line
        if lines:
            if lines[0].startswith("```"):
                lines = lines[1:]
        # remove trailing fence line
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s


def _best_effort_json_parse(raw_text: str) -> Dict[str, Any]:
    """
    Try very hard to parse model output as JSON:
    - strip code fences
    - cut to first '{' ... last '}'
    - remove obvious trailing commas
    - balance braces/brackets if needed
    """
    cleaned = _strip_code_fences(raw_text)

    # Focus on main JSON object segment
    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first != -1 and last != -1 and last > first:
        cleaned = cleaned[first:last + 1]

    # Remove trailing commas before closing } or ]
    cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)

    # Balance braces / brackets if model truncated slightly
    open_curly = cleaned.count("{")
    close_curly = cleaned.count("}")
    open_sq = cleaned.count("[")
    close_sq = cleaned.count("]")

    if close_curly < open_curly:
        cleaned += "}" * (open_curly - close_curly)
    if close_sq < open_sq:
        cleaned += "]" * (open_sq - close_sq)

    # Final parse
    return json.loads(cleaned)


def call_groq_for_page(
    page_no: int,
    image_data_url: str,
    ocr_text: str,
    max_retries: int = 2,
) -> Tuple[Dict[str, Any], TokenUsage]:
    """
    Call Groq Vision model for a single page image, with OCR text as hint.
    Uses retries + best-effort JSON repair for robustness.
    """
    # truncate OCR snippet
    ocr_text_snippet = (ocr_text or "").strip()
    if len(ocr_text_snippet) > 1500:
        ocr_text_snippet = ocr_text_snippet[:1500]

    base_user_text = f"""
You are processing page number {page_no} of a multi-page medical bill.

Use page_no="{page_no}" in the JSON.
Classify page_type as exactly one of:
- "Bill Detail"
- "Final Bill"
- "Pharmacy"

Below is OCR text extracted from THIS PAGE ONLY (may be noisy).
Use it only as a helper; always respect the actual table structure.

OCR_TEXT_START
{ocr_text_snippet}
OCR_TEXT_END

Remember:
- Output ONLY items that visibly correspond to table rows on THIS PAGE.
- DO NOT include totals / subtotals / GST / discounts / net amounts.
- DO NOT invent items from other pages.
- Output STRICT JSON, no markdown, no comments.
"""

    total_tokens = 0
    input_tokens = 0
    output_tokens = 0
    last_error = None

    for attempt in range(max_retries + 1):
        if attempt == 0:
            user_text = base_user_text
        else:
            # Retry with an explicit correction instruction
            user_text = (
                base_user_text
                + "\n\nYour previous response was not valid JSON. "
                  "Now respond ONLY with a single valid JSON object, "
                  "with balanced brackets and no markdown fences."
            )

        try:
            response = client.responses.create(
                model=GROQ_VISION_MODEL_ID,
                input=[
                    {
                        "role": "system",
                        "content": [
                            {"type": "input_text", "text": SYSTEM_PROMPT.strip()},
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": user_text.strip()},
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
            last_error = f"Groq API error: {e}"
            continue

        usage = getattr(response, "usage", None)
        if usage is not None:
            total_tokens += int(getattr(usage, "total_tokens", 0) or 0)
            input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
            output_tokens += int(getattr(usage, "output_tokens", 0) or 0)

        raw_text = response.output_text or ""
        raw_text = raw_text.strip()

        try:
            parsed = _best_effort_json_parse(raw_text)
            token_usage = TokenUsage(
                total_tokens=total_tokens,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            return parsed, token_usage
        except Exception as e:
            last_error = f"JSON parse error: {e} | raw_head={raw_text[:200]!r}"
            # loop and retry

    # all attempts failed
    raise HTTPException(
        status_code=500,
        detail=f"Model failed to return valid JSON after retries: {last_error}",
    )


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


_TOTAL_PHRASES = [
    "sub total",
    "subtotal",
    "grand total",
    "net amount",
    "net amt",
    "net payable",
    "net pay",
    "total amount",
    "total bill",
    "total charges",
    "total payable",
    "total rs",
    "round off",
    "round-off",
    "discount",
    "concession",
    "gst",
    "cgst",
    "sgst",
    "igst",
    "tax",
]


def _looks_like_pure_total_row(name: str) -> bool:
    """
    Decide if a row is a pure total / summary row that should be removed.
    We are careful to NOT drop genuine tests like 'Total Protein'.
    """
    lower = name.lower().strip()

    # If it's just 'total', 'grand total', etc.
    if lower in {
        "total",
        "grand total",
        "sub total",
        "subtotal",
        "total amount",
        "net amount",
        "net amount payable",
        "total bill",
        "total charges",
    }:
        return True

    # If it contains strong multi-word total phrases
    for phrase in _TOTAL_PHRASES:
        if phrase in lower:
            # But ignore common clinical test patterns
            if "protein" in lower or "bilirubin" in lower:
                continue
            return True

    return False


def clean_page_dict(page_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean raw JSON from model so Pydantic parsing won't fail.
    - Coerce numeric fields.
    - Remove within-page duplicate rows.
    - Drop obvious summary/total rows.
    """
    bill_items = page_dict.get("bill_items", []) or []
    cleaned_items: List[Dict[str, Any]] = []

    seen_keys: Set[Tuple[str, float, float, float]] = set()

    for item in bill_items:
        if not isinstance(item, dict):
            continue

        name = str(item.get("item_name", "")).strip()
        if not name:
            continue

        # Drop obvious totals / summaries
        if _looks_like_pure_total_row(name):
            continue

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
            # LLM hallucinated a duplicate of same row in this page
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

    page_type = str(page_dict.get("page_type", "Bill Detail") or "Bill Detail")
    page_no = str(page_dict.get("page_no", "") or "")

    return {
        "page_no": page_no,
        "page_type": page_type,
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

    We treat items with identical:
        (page_type, normalized_name, rate, qty, amount)
    as duplicates and keep only the first.
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
                continue
            seen.add(key)
            new_items.append(item)

        p.bill_items = new_items
        deduped_pages.append(p)

    return deduped_pages


def aggregate_all_pages(pages: List[PageItems]) -> ExtractBillDataResponseData:
    """
    Compute total_item_count.
    Grand total is NOT part of schema, but we log it for debugging.
    """
    total_items = sum(len(p.bill_items) for p in pages)
    grand_total = sum(float(it.item_amount) for p in pages for it in p.bill_items)

    print(
        f"[AGG] pages={len(pages)} total_item_count={total_items} "
        f"grand_total={grand_total:.2f}"
    )

    return ExtractBillDataResponseData(
        pagewise_line_items=pages,
        total_item_count=total_items,
    )


# ============================================================
#  API Endpoints
# ============================================================

@app.get("/extract-bill-data")
def health_check():
    """
    Simple health-check for evaluators that send GET.
    Does NOT perform extraction; just tells how to use POST.
    """
    return {
        "message": (
            "Bill extraction API is healthy. "
            "Use POST /extract-bill-data with JSON body "
            "{\"document\": \"<public document URL>\"}."
        )
    }


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
          "token_usage": { ... },
          "data": {
            "pagewise_line_items": [...],
            "total_item_count": int
          }
        }
    """
    url_str = str(req.document)
    print(f"[REQ] document={url_str}")

    # 1. Download
    content = download_document(url_str)

    # 2. Split into per-page infos
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

    logical_page_no = 1

    # 3. Process each non-blank page
    for info in page_infos:
        if info.get("is_blank"):
            print(f"[PAGE {logical_page_no}] skipped as blank")
            logical_page_no += 1
            continue

        data_url = info["data_url"]
        ocr_text = info.get("ocr_text", "") or ""

        parsed_json, usage = call_groq_for_page(
            page_no=logical_page_no,
            image_data_url=data_url,
            ocr_text=ocr_text,
        )

        page_items = reconcile_page_items(parsed_json)
        page_items.page_no = str(logical_page_no)
        all_pages.append(page_items)

        total_tokens += usage.total_tokens
        input_tokens += usage.input_tokens
        output_tokens += usage.output_tokens

        print(
            f"[PAGE {logical_page_no}] type={page_items.page_type} "
            f"items={len(page_items.bill_items)} "
            f"tokens(total/in/out)={usage.total_tokens}/"
            f"{usage.input_tokens}/{usage.output_tokens}"
        )

        logical_page_no += 1

    if not all_pages:
        return ExtractBillDataResponse(
            is_success=False,
            message="All pages were detected as blank; nothing to extract.",
        )

    # 4. Cross-page dedupe
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
