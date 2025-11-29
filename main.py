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
import pytesseract  # OCR for high recall on tables

# ============================================================
#  Groq Client (OpenAI-compatible)
# ============================================================

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

# High-accuracy vision model (Maverick) – default
GROQ_VISION_MODEL_ID = os.environ.get(
    "GROQ_VISION_MODEL_ID",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
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
        "High-accuracy extraction of line items from multi-page hospital bills "
        "using OCR + Groq LLaMA 4 Maverick vision in batched mode. "
        "Implements the exact HackRx Datathon schema."
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
    Light contrast + sharpness boost to help faint numeric columns.
    """
    img = ImageEnhance.Contrast(img).enhance(1.6)
    img = ImageEnhance.Sharpness(img).enhance(1.4)
    return img


def _ocr_page(img: Image.Image) -> str:
    """
    Run Tesseract OCR on a page to get raw text lines.
    This greatly increases recall of all table rows.
    """
    try:
        gray = img.convert("L")
        gray = ImageEnhance.Contrast(gray).enhance(1.8)
        text = pytesseract.image_to_string(
            gray,
            config="--psm 6"  # Assume a single uniform block of text (tables)
        )
        return text
    except Exception:
        # Fallback: no OCR, model will rely on vision only
        return ""


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
          "data_url": "data:image/jpeg;base64,...",
          "ocr_text": "<raw OCR text of the page>"
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

        ocr_text = _ocr_page(img)
        page_infos.append(
            {
                "page_index": 0,
                "data_url": image_to_data_url(img),
                "ocr_text": ocr_text,
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
            ocr_text = _ocr_page(img)
            page_infos.append(
                {
                    "page_index": idx,
                    "data_url": image_to_data_url(img),
                    "ocr_text": ocr_text,
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

    ocr_text = _ocr_page(img)
    page_infos.append(
        {
            "page_index": 0,
            "data_url": image_to_data_url(img),
            "ocr_text": ocr_text,
        }
    )
    return page_infos


# ============================================================
#  LLM Prompt – batched, OCR + vision, strict JSON
# ============================================================

SYSTEM_PROMPT = """
You are an expert medical BILL ITEM extraction engine for hospital bills.

Your goals:

1) Capture EVERY genuine line item from the tables (high recall).
2) Do NOT double-count or duplicate the same line item.
3) Make the sum of all `item_amount` values across all pages as close as possible
   to the FINAL TOTAL printed in the bill.
4) Respect the exact JSON structure required by the HackRx Datathon.

You are given, for each page:
- Full OCR TEXT of that page (primary source of truth).
- The page IMAGE (for visual confirmation only).

CRITICAL BEHAVIOUR:

- Treat the OCR text lines as the canonical rows of the table.
- For EACH visual/OCR row that describes a charge, output exactly ONE bill item.
- Do NOT merge multiple rows into one item, even if the service name is similar.
- Do NOT invent new rows which do not exist in the OCR text.

- Section totals such as "TOTAL", "SUB TOTAL", "SUB-TOTAL",
  "GRAND TOTAL", "NET AMOUNT PAYABLE", "NET PAYABLE", "ROUND OFF",
  "DISCOUNT", etc. MUST NOT be emitted as bill_items.

NUMERIC RULES:

- item_quantity: read from "Qty", "No. of Days", or similar columns.
- item_rate:     from "Rate", "Charges per day", etc.
- item_amount:   from "Amount", "Net Amt", "Company Amount", etc.
- If there are multiple numeric columns, choose the column that
  represents the NET amount being charged in that row.

If some numeric fields are missing for a row BUT other rows with the
same service name have clear numbers, you may infer the missing numbers
as long as they are consistent with the pattern (e.g. same rate per unit).
Otherwise, set unknown numeric fields to 0.0.

OUTPUT FORMAT (PER BATCH):

You will see K page images and OCR texts in this batch, in order.
For EACH page i in this batch (1-based):

- Decide page_type ∈ {"Bill Detail", "Final Bill", "Pharmacy"}.
- Extract ALL line items for that page (one per genuine charge row).

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
- Do NOT output totals, sub-totals, taxes, discounts, round-off,
  or any summary-only rows as bill_items.
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
You are given a BATCH of {num_batch_pages} page(s) from a hospital bill.

For EACH batch page i (1-based), you receive:
1) OCR TEXT for that page.
2) The page IMAGE for confirmation.

Use the OCR TEXT as the primary source of rows and numbers, and use the
image only to resolve ambiguities.

For EACH batch page i you must:
- Set page_no = "<i>" (as a string)
- Choose page_type from: "Bill Detail", "Final Bill", "Pharmacy"
- Extract bill_items ONLY for that page.
"""

    user_content: List[Dict[str, Any]] = [
        {"type": "input_text", "text": user_text.strip()}
    ]

    for idx, info in enumerate(batch_page_infos):
        batch_page_no = idx + 1
        ocr_text = (info.get("ocr_text") or "").strip()

        user_content.append(
            {
                "type": "input_text",
                "text": f"=== BATCH PAGE {batch_page_no} OCR TEXT START ===\n"
                        f"{ocr_text}\n"
                        f"=== BATCH PAGE {batch_page_no} OCR TEXT END ===",
            }
        )
        user_content.append(
            {
                "type": "input_text",
                "text": f"Image for BATCH PAGE {batch_page_no} is below.",
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

    # Robust JSON parsing (strip fences if any)
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

    raw_pages = raw_pages[:num_batch_pages]
    return raw_pages, token_usage


# ============================================================
#  Reconcile, Clean, Deduplicate & Aggregate
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


SUMMARY_KEYWORDS = (
    "TOTAL",
    "SUB TOTAL",
    "SUB-TOTAL",
    "GRAND TOTAL",
    "NET AMOUNT",
    "NET PAYABLE",
    "NET AMT",
    "ROUND OFF",
    "ROUND-OFF",
    "ROUND OFF AMT",
    "DISCOUNT",
)


def clean_page_dict(page_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pre-clean the raw JSON dict from the model so that Pydantic parsing will not
    fail when numeric fields are null/empty/etc.

    We also drop obvious summary/total rows by keywords.
    """
    bill_items = page_dict.get("bill_items", []) or []
    cleaned_items: List[Dict[str, Any]] = []

    for item in bill_items:
        if not isinstance(item, dict):
            continue

        name = str(item.get("item_name", "")).strip()
        if not name:
            continue

        upper_name = name.upper()
        # Drop obvious totals / summary lines
        if any(k in upper_name for k in SUMMARY_KEYWORDS):
            continue

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


def dedupe_with_ocr(page: PageItems, ocr_text: str) -> PageItems:
    """
    Use OCR text to drop hallucinated duplicate rows.

    Idea:
      - Count how many times each item_name appears in OCR text.
      - Model is not allowed to output that item more times than OCR count.
      - Keep the first N occurrences in original order.
    """
    if not ocr_text:
        return page

    text_lower = ocr_text.lower()
    name_counts_model: Dict[str, int] = defaultdict(int)
    for item in page.bill_items:
        key = item.item_name.strip().lower()
        name_counts_model[key] += 1

    allowed_counts: Dict[str, int] = {}
    for name, model_count in name_counts_model.items():
        occ = text_lower.count(name)
        if occ <= 0:
            # OCR may be noisy; in that case, trust the model count.
            allowed_counts[name] = model_count
        else:
            allowed_counts[name] = min(model_count, occ)

    seen: Dict[str, int] = defaultdict(int)
    new_items: List[BillItem] = []
    for item in page.bill_items:
        key = item.item_name.strip().lower()
        if seen[key] < allowed_counts.get(key, 0):
            new_items.append(item)
            seen[key] += 1
        else:
            # extra hallucinated duplicate – drop it
            continue

    page.bill_items = new_items
    return page


def enrich_from_patterns(pages: List[PageItems]) -> List[PageItems]:
    """
    SECOND PASS (conservative):
    If some rows have missing numeric fields, but other rows with the same
    item_name have a SINGLE consistent (rate, qty) pattern, use that to fill.

    This is intentionally conservative to avoid corrupting correct values.
    """
    # 1) Collect per-name statistics
    patterns: Dict[str, set] = defaultdict(set)  # name -> set of (rate, qty)
    for p in pages:
        for it in p.bill_items:
            name_key = it.item_name.strip().lower()
            amt = float(it.item_amount)
            rate = float(it.item_rate)
            qty = float(it.item_quantity)

            if rate > 0 and qty > 0:
                patterns[name_key].add((round(rate, 4), round(qty, 4)))
            elif amt > 0 and qty > 0:
                inferred_rate = amt / qty
                patterns[name_key].add((round(inferred_rate, 4), round(qty, 4)))

    # 2) Build defaults only when there is a SINGLE consistent pattern
    defaults: Dict[str, Tuple[float, float]] = {}
    for name, pats in patterns.items():
        if len(pats) == 1:
            r, q = list(pats)[0]
            if r > 0 and q > 0:
                defaults[name] = (r, q)

    # 3) Fill missing fields using defaults
    for p in pages:
        for it in p.bill_items:
            name_key = it.item_name.strip().lower()
            amt = float(it.item_amount)
            rate = float(it.item_rate)
            qty = float(it.item_quantity)

            default = defaults.get(name_key)
            if not default:
                # No reliable pattern; leave as is
                it.item_quantity = float(qty if qty > 0 else 1.0)
                it.item_rate = round(float(rate), 2) if rate > 0 else 0.0
                it.item_amount = round(float(amt), 2) if amt > 0 else 0.0
                continue

            default_rate, default_qty = default

            # Only modify fields that are missing / zero
            if qty <= 0:
                qty = default_qty
            if rate <= 0 and amt > 0 and qty > 0:
                rate = amt / qty
            if rate <= 0:
                rate = default_rate
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
#  Health Check (GET on same endpoint)
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
    url_str = str(req.document)

    # 1. Download document
    content = download_document(url_str)

    # 2. Convert to per-page infos (image → data_url + ocr_text)
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

        # Map batch-local page_no → global page_no and apply OCR-based dedup
        for i, page_dict in enumerate(raw_pages_batch):
            global_page_no = batch_start + i + 1  # 1-based for entire document
            page_dict["page_no"] = str(global_page_no)

            page_items = reconcile_page_items(page_dict)

            # OCR-based dedup for this page
            ocr_text = batch_page_infos[i].get("ocr_text") or ""
            page_items = dedupe_with_ocr(page_items, ocr_text)

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

    # 4. Enrich missing numeric fields using conservative global patterns
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
