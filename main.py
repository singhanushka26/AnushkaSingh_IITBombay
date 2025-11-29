# main.py
# Hybrid Tesseract OCR + Groq Maverick
# - Per-page OCR
# - Text-only LLM parsing
# - Robust JSON repair
# - Summary filtering + dedupe + pattern enrichment

import base64
import io
import json
import math
import mimetypes
import os
import time
from collections import defaultdict
from typing import List, Optional, Dict, Any, Tuple

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from pdf2image import convert_from_bytes
from PIL import Image, ImageEnhance, ImageOps
from openai import OpenAI
import pytesseract
from pytesseract import Output

# ============================================================
#  Groq Client (OpenAI-compatible)
# ============================================================

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

GROQ_VISION_MODEL_MAVERICK = os.environ.get(
    "GROQ_VISION_MODEL_MAVERICK",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
)

# ============================================================
#  Datathon Schemas
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
    title="Bajaj Datathon Bill Extraction API (OCR + Maverick)",
    version="1.0.0",
    description=(
        "Hybrid pipeline using Tesseract OCR + Groq Maverick to extract "
        "bill line items from multi-page medical bills, following HackRx schema."
    ),
)

# ============================================================
#  Download & Document → Page Images
# ============================================================

def download_document(url: str) -> bytes:
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
    mime, _ = mimetypes.guess_type(url)
    if mime:
        return mime
    if content[:4] == b"%PDF":
        return "application/pdf"
    return "application/octet-stream"


def preprocess_for_ocr(img: Image.Image) -> Image.Image:
    """
    Preprocessing tuned for Tesseract:
    - Slight upscale
    - Grayscale
    - Contrast & sharpness boost
    """
    # Upscale a bit for small fonts
    w, h = img.size
    scale = 1.3
    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    img = ImageOps.grayscale(img)
    img = ImageEnhance.Contrast(img).enhance(1.4)
    img = ImageEnhance.Sharpness(img).enhance(1.2)
    return img


def document_to_pil_pages(url: str, content: bytes) -> List[Image.Image]:
    mime = guess_mime_type(url, content)

    if mime.startswith("image/"):
        img = Image.open(io.BytesIO(content)).convert("RGB")
        return [img]

    if mime == "application/pdf":
        try:
            # 300 dpi gives better OCR quality without being too heavy
            pages = convert_from_bytes(content, dpi=300)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Unable to convert PDF to images (poppler needed): {e}",
            )
        return [p.convert("RGB") for p in pages]

    # Fallback: try as image
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
        return [img]
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported document type: {mime}",
        )


# ============================================================
#  OCR: Page → Text Lines (rows)
# ============================================================

def ocr_page_to_lines(img: Image.Image) -> List[str]:
    """
    Run Tesseract OCR on a single page and return a list of
    text lines in reading order. Each line is a candidate table row.
    """
    img_proc = preprocess_for_ocr(img)

    data = pytesseract.image_to_data(
        img_proc,
        output_type=Output.DICT,
        config="--oem 3 --psm 6",
    )

    n = len(data["text"])
    line_map: Dict[Tuple[int, int, int], List[str]] = {}

    for i in range(n):
        text = data["text"][i].strip()
        if not text:
            continue

        conf_str = data["conf"][i]
        try:
            conf = int(conf_str)
        except ValueError:
            conf = -1

        # Filter very low-confidence garbage
        if conf < 40:
            continue

        block = data["block_num"][i]
        par = data["par_num"][i]
        line = data["line_num"][i]
        key = (block, par, line)

        line_map.setdefault(key, []).append(text)

    # Sort by (block, par, line) to preserve reading order
    rows: List[str] = []
    for key in sorted(line_map.keys()):
        line_text = " ".join(line_map[key]).strip()
        # Skip very short junk
        if len(line_text) < 3:
            continue
        rows.append(line_text)

    return rows


# ============================================================
#  LLM Prompt + JSON Extractor
# ============================================================

SYSTEM_PROMPT_MAVERICK = """
You are an expert medical bill parser.

You receive OCR-extracted text LINES from ONE page of a hospital bill.
Each line is in top-to-bottom reading order and may be:
- a table header,
- a section title,
- a detailed charge row,
- a summary row (TOTAL / SUB TOTAL / NET AMOUNT / ROUND OFF / DISCOUNT / BALANCE),
- or unrelated text (patient details, address, etc.).

YOUR TASK FOR THIS SINGLE PAGE:

1. Decide the page_type for this page:
   - "Bill Detail" if it mainly contains detailed service/room/doctor charges.
   - "Pharmacy" if it mainly contains medicine or pharmacy items.
   - "Final Bill" if it mainly contains final summary/final total/bill overview.

2. From ONLY the lines representing detailed charge items, build bill_items.
   - One bill_items entry per detailed row.
   - Do NOT create items from header rows or summary rows.

3. For each bill_item:
   - item_name: full description of the service/charge (as close to OCR text as possible).
   - item_quantity: numeric quantity (Qty, No. of days, units, etc.). If missing, use 1.0.
   - item_rate: per-unit rate or charge. If missing but amount & qty present, infer rate = amount/qty.
   - item_amount: final net amount for that row. If missing but rate & qty known, infer amount = rate*qty.

4. Very important:
   - NEVER output totals, sub-totals, grand totals, net amount payable, round-off, discounts, or balance rows as bill_items.
   - If a line clearly is a summary (contains words like TOTAL, SUBTOTAL, NET, GRAND, ROUND, DISCOUNT, BALANCE, etc.), ignore it.
   - If numbers are unreadable, set them to 0.0 rather than guessing large values.

OUTPUT FORMAT (JSON only, no markdown, no comments):

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

- page_no must be "1" for this single-page call.
- total_item_count = number of bill_items on this page.
- JSON MUST be valid. No trailing commas. No extra top-level keys.
"""


def safe_json_extract(raw: str) -> Optional[Dict[str, Any]]:
    """
    Try very hard to recover a JSON object from raw LLM text.
    Returns dict or None if completely impossible.
    """
    raw = raw.replace("```json", "").replace("```", "").strip()

    first = raw.find("{")
    last = raw.rfind("}")

    if first == -1:
        return None

    if last == -1 or last <= first:
        text = raw[first:]
    else:
        text = raw[first:last+1]

    # Basic cleanup
    text = text.replace(",}", "}").replace(",]", "]")

    # Brace balancing
    need_curly = text.count("{") - text.count("}")
    if need_curly > 0:
        text += "}" * need_curly

    need_square = text.count("[") - text.count("]")
    if need_square > 0:
        text += "]" * need_square

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try trimming trailing garbage
        for cut in range(len(text)-1, 0, -1):
            try:
                return json.loads(text[:cut])
            except Exception:
                continue

    return None


def call_maverick_for_page(
    page_idx: int,
    ocr_lines: List[str],
) -> Tuple[Dict[str, Any], TokenUsage]:
    """
    Single-page call to Maverick:
    - Input: OCR lines from that page
    - Output: raw page dict + token usage
    """

    if not ocr_lines:
        # No text; return empty page
        empty_page = {
            "page_no": "1",
            "page_type": "Bill Detail",
            "bill_items": [],
        }
        return empty_page, TokenUsage(total_tokens=0, input_tokens=0, output_tokens=0)

    lines_text = "\n".join(
        f"{i+1}. {line}" for i, line in enumerate(ocr_lines)
    )

    user_prompt = f"""
You are given OCR-extracted lines from ONE page of a hospital bill.

This is PAGE {page_idx + 1} of the full document.
Lines are in reading order (top to bottom):

{lines_text}

Now produce the JSON for this SINGLE PAGE exactly as per the schema.
Remember:
- page_no must be "1" in your JSON (we will remap it later).
- Exclude summary/total/discount rows from bill_items.
"""

    try:
        response = client.responses.create(
            model=GROQ_VISION_MODEL_MAVERICK,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": SYSTEM_PROMPT_MAVERICK.strip()}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_prompt.strip()}],
                },
            ],
        )
    except Exception as e:
        # On LLM failure, return empty page instead of 500
        print(f"[MAVERICK_ERROR] page={page_idx+1} error={e}")
        empty_page = {
            "page_no": "1",
            "page_type": "Bill Detail",
            "bill_items": [],
        }
        return empty_page, TokenUsage(total_tokens=0, input_tokens=0, output_tokens=0)

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
    parsed = safe_json_extract(raw_text)

    if not parsed:
        print(f"[MAVERICK_JSON_FAIL] page={page_idx+1} text_snip={raw_text[:200]}")
        empty_page = {
            "page_no": "1",
            "page_type": "Bill Detail",
            "bill_items": [],
        }
        return empty_page, token_usage

    if isinstance(parsed, dict) and "pagewise_line_items" in parsed:
        pw = parsed.get("pagewise_line_items", []) or []
    elif isinstance(parsed, list):
        pw = parsed
    else:
        pw = []

    if not pw:
        empty_page = {
            "page_no": "1",
            "page_type": "Bill Detail",
            "bill_items": [],
        }
        return empty_page, token_usage

    return pw[0], token_usage


# ============================================================
#  Cleaning, Filtering, Enrichment
# ============================================================

SUMMARY_WORDS = ("TOTAL", "SUB", "NET", "GRAND", "ROUND", "BALANCE", "DISCOUNT")


def _coerce_number(x: Any) -> float:
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
    bill_items = page_dict.get("bill_items", []) or []
    cleaned_items: List[Dict[str, Any]] = []

    for item in bill_items:
        if not isinstance(item, dict):
            continue

        name = str(item.get("item_name", "")).strip()
        # Drop summary/total style rows as extra safety
        if any(w in name.upper() for w in SUMMARY_WORDS):
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


def dedupe_page(p: PageItems) -> PageItems:
    seen = set()
    final_items: List[BillItem] = []
    for it in p.bill_items:
        key = (it.item_name.lower(), it.item_amount, it.item_rate, it.item_quantity)
        if key in seen:
            continue
        seen.add(key)
        final_items.append(it)
    p.bill_items = final_items
    return p


def enrich_from_patterns(pages: List[PageItems]) -> List[PageItems]:
    """
    If some rows have missing numeric fields, but other rows with the same
    item_name have a consistent (rate, qty) pattern, use that to fill in.
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

    default_rate = {k: (sum(v) / len(v)) for k, v in rates.items() if v}
    default_qty = {k: (sum(v) / len(v)) for k, v in qtys.items() if v}

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
#  Health Check
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

    # 1. Download
    content = download_document(url_str)

    # 2. PDF/image → PIL pages
    pil_pages = document_to_pil_pages(url_str, content)
    num_pages = len(pil_pages)

    if num_pages == 0:
        elapsed = time.time() - start_time
        print(f"[BILL_EXTRACT] pages=0 items=0 total_amount=0.00 tokens=0 time_sec={elapsed:.2f} (no pages)")
        return ExtractBillDataResponse(
            is_success=False,
            message="No pages/images could be extracted from the document.",
        )

    all_pages: List[PageItems] = []
    total_tokens = 0
    input_tokens = 0
    output_tokens = 0

    # 3. For EACH page: OCR → Maverick → PageItems
    for page_idx, pil_page in enumerate(pil_pages):
        ocr_lines = ocr_page_to_lines(pil_page)
        raw_page, usage = call_maverick_for_page(page_idx, ocr_lines)

        total_tokens += usage.total_tokens
        input_tokens += usage.input_tokens
        output_tokens += usage.output_tokens

        # Remap page_no from local "1" to global page index
        raw_page["page_no"] = str(page_idx + 1)

        page_items = reconcile_page_items(raw_page)
        page_items = dedupe_page(page_items)

        all_pages.append(page_items)

    if not all_pages:
        elapsed = time.time() - start_time
        print(f"[BILL_EXTRACT] pages={num_pages} items=0 total_amount=0.00 tokens={total_tokens} time_sec={elapsed:.2f} (no items)")
        return ExtractBillDataResponse(
            is_success=False,
            token_usage=TokenUsage(
                total_tokens=total_tokens,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
            message="Models returned no page items.",
        )

    # 4. Enrich missing numeric fields
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
