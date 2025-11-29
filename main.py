# main.py
import base64
import io
import json
import math
import mimetypes
import os
import re
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

MODEL_FAST = "meta-llama/llama-4-scout-17b-16e-instruct"
MODEL_ACCURATE = "meta-llama/llama-4-maverick-17b-128e-instruct"

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
    version="6.1.0",
    description=(
        "Hybrid Scout + Maverick pipeline with batching (<=5 images per "
        "request) and suspicious-page recovery for high accuracy while "
        "staying within latency limits. Includes robust JSON repair."
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


def _resize_for_vision(img: Image.Image, max_dim: int = 1100) -> Image.Image:
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


def image_to_data_url(img: Image.Image, quality: int = 65) -> str:
    """
    Convert a PIL Image into a base64 JPEG data URL.

    - Resize to max_dim=1100
    - JPEG quality ~65 (slightly higher than before for better accuracy)
    - Ensure < 4 MB per image as per Groq base64 limits.
    """
    img = _resize_for_vision(img, max_dim=1100)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b = buf.getvalue()

    while len(b) > 4 * 1024 * 1024 and quality > 35:
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
            # Slightly lower dpi for speed; still enough for tables.
            pages = convert_from_bytes(content, dpi=170)
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
#  Prompts – Multi-page (batched) & single-page recovery
# ============================================================

SYSTEM_PROMPT_BASE = """
You are an expert medical BILL ITEM extraction engine for hospital bills.

GLOBAL GOALS:
1) Capture EVERY genuine line item (high recall).
2) Do NOT double-count or duplicate the same line item.
3) Make the sum of all `item_amount` values close to the FINAL TOTAL on the bill.
4) Respect the exact JSON structure required by the HackRx Datathon.
5) Output ONLY JSON. No explanations, no markdown, no backticks.
"""

USER_PROMPT_BATCH = """
You will receive a BATCH of page images from a hospital bill.

For EACH page in this batch:
- Identify page_type ∈ {"Bill Detail", "Final Bill", "Pharmacy"}.
- Extract only REAL line items (description + amount) from table rows.
- Do NOT output header/title rows.
- Do NOT output total/sub-total/tax/round-off rows.
- Do NOT create two JSON items for a single visual row.

For numeric fields:
- If quantity is missing but amount is present: quantity = 1.0 and rate = amount.
- If rate is missing but quantity and amount are present: rate = amount / quantity.
- If any numeric is unreadable: use 0.0 (not null / empty string).
- Use JSON numbers only (e.g. 1200.5, not "1,200.5").

Return ONE strict JSON object ONLY:

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
      "page_type": "...",
      "bill_items": [ ... ]
    }
  ],
  "total_item_count": <int>
}

Where page_no is the 1-based index WITHIN THIS BATCH, as a STRING.
"""

USER_PROMPT_SINGLE = """
You are re-checking ONE SINGLE PAGE of a hospital bill.

- Carefully read all table rows with charges.
- Extract EVERY line item: description + quantity + rate + amount.
- Do NOT output headers or section titles.
- Do NOT output totals, sub-totals, taxes, discounts, or round-off rows.
- Do NOT double-count rows: exactly one JSON item per visual row.

Return ONE strict JSON object ONLY:

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
  "total_item_count": <int>
}
"""


# ============================================================
#  JSON Repair + Parsing
# ============================================================

def _repair_json_string(s: str) -> str:
    """
    Try to fix common LLM JSON mistakes:

    - Numbers like 3000.  -> 3000.0
    - Trailing commas before } or ]
    - Strip stray leading/trailing text outside the outermost { ... }
    """
    txt = s.strip()

    # Keep only the substring from first '{' to last '}'
    first = txt.find("{")
    last = txt.rfind("}")
    if first != -1 and last != -1 and last > first:
        txt = txt[first:last + 1]

    # Fix numbers like 3000. (no digit after the dot)
    # Pattern: integer followed by '.' not followed by a digit
    txt = re.sub(r'(?P<num>-?\d+)\.(?=\D)', r'\g<num>.0', txt)

    # Remove trailing commas before } or ]
    txt = re.sub(r',(\s*[\]}])', r'\1', txt)

    return txt


def _parse_groq_json(raw_text: str) -> dict:
    raw = raw_text.strip()

    # Strip accidental markdown fences
    if raw.startswith("```"):
        # Remove leading and trailing ``` blocks
        raw = raw.strip("`")

    # First attempt: direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try with repair
        repaired = _repair_json_string(raw)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=500,
                detail=f"Model response is not valid JSON: {repaired[:200]}",
            )


# ============================================================
#  Groq Call Helpers
# ============================================================

def call_groq_for_batch_fast(
    batch_page_infos: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], TokenUsage]:
    """
    Single Groq Vision call (FAST model) for a batch of pages (<=5 images).
    Returns:
        - list of per-page dicts (page_no, page_type, bill_items)
        - token usage for this batch
    """
    num_batch_pages = len(batch_page_infos)

    user_text = f"""
You are given a BATCH of {num_batch_pages} page image(s) from a hospital bill.

Images are in order for this batch:
1st image = batch page 1, 2nd = batch page 2, ..., {num_batch_pages}th = batch page {num_batch_pages}.

Follow the instructions in the USER_PROMPT strictly and output ONLY JSON.
"""

    user_content: List[Dict[str, Any]] = [
        {"type": "input_text", "text": USER_PROMPT_BATCH.strip()},
        {"type": "input_text", "text": user_text.strip()},
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
            model=MODEL_FAST,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": SYSTEM_PROMPT_BASE.strip()}],
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
            detail=f"Groq API error (fast batch): {e}",
        )

    usage = getattr(response, "usage", None)
    tu = TokenUsage(
        total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
    )

    parsed = _parse_groq_json(response.output_text)

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

    return raw_pages[:num_batch_pages], tu


def call_groq_single_page_accurate(
    page_info: Dict[str, Any],
) -> Tuple[Dict[str, Any], TokenUsage]:
    """
    Single-page high-accuracy pass with Maverick.
    Returns a dict representing one page (page_no, page_type, bill_items).
    """
    user_content: List[Dict[str, Any]] = [
        {"type": "input_text", "text": USER_PROMPT_SINGLE.strip()},
        {
            "type": "input_image",
            "image_url": page_info["data_url"],
            "detail": "high",
        },
    ]

    try:
        response = client.responses.create(
            model=MODEL_ACCURATE,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": SYSTEM_PROMPT_BASE.strip()}],
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
            detail=f"Groq API error (accurate page): {e}",
        )

    usage = getattr(response, "usage", None)
    tu = TokenUsage(
        total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
    )

    parsed = _parse_groq_json(response.output_text)

    if isinstance(parsed, dict) and "pagewise_line_items" in parsed:
        plist = parsed.get("pagewise_line_items", []) or []
        if not plist:
            return {
                "page_no": "1",
                "page_type": "Bill Detail",
                "bill_items": [],
            }, tu
        return plist[0], tu

    # If model returned a single page dict directly
    if isinstance(parsed, dict) and "bill_items" in parsed:
        return parsed, tu

    raise HTTPException(
        status_code=500,
        detail="High-accuracy model JSON is not in expected format.",
    )


# ============================================================
#  Cleaning & Normalisation
# ============================================================

def _coerce_number(x: Any) -> float:
    """Coerce a potentially messy numeric value to float."""
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
    Pre-clean raw JSON dict from the model so Pydantic parsing will not fail.
    Also removes exact duplicates WITHIN the page.
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
    """Validate and normalise a single page's JSON dict into PageItems."""
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
    """Compute total_item_count."""
    total_items = sum(len(p.bill_items) for p in pages)
    return ExtractBillDataResponseData(
        pagewise_line_items=pages,
        total_item_count=total_items,
    )


def compute_grand_total_amount(pages: List[PageItems]) -> float:
    """Utility: sum of all item_amounts – for logging only."""
    total = 0.0
    for p in pages:
        for item in p.bill_items:
            total += float(item.item_amount)
    return round(total, 2)


def find_suspicious_pages(pages: List[PageItems]) -> List[int]:
    """
    Heuristics to decide which pages need high-accuracy re-check.

    - No items OR <= 2 items.
    - Or one item_name dominating (>70% of rows, and appears >=4 times).
    """
    suspicious: List[int] = []

    for idx, p in enumerate(pages):
        n = len(p.bill_items)
        if n == 0 or n <= 2:
            suspicious.append(idx)
            continue

        # dominance of a single name (e.g., many identical NICU rows)
        name_counts: Dict[str, int] = {}
        for it in p.bill_items:
            key = it.item_name.strip().lower()
            if not key:
                continue
            name_counts[key] = name_counts.get(key, 0) + 1

        if not name_counts:
            suspicious.append(idx)
            continue

        max_count = max(name_counts.values())
        if max_count >= 4 and max_count >= 0.7 * n:
            suspicious.append(idx)

    # de-duplicate & keep sorted
    return sorted(set(suspicious))


# ============================================================
#  Health Check (GET)
# ============================================================

@app.get("/extract-bill-data")
def health_check():
    """
    Simple GET endpoint so health checks don't see 405.
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

    # 3. FAST PASS: process pages in batches (Scout)
    all_pages_fast: List[PageItems] = []
    total_tokens = 0
    input_tokens = 0
    output_tokens = 0

    for batch_start in range(0, num_pages, MAX_IMAGES_PER_REQUEST):
        batch_end = min(batch_start + MAX_IMAGES_PER_REQUEST, num_pages)
        batch_page_infos = page_infos[batch_start:batch_end]

        raw_pages_batch, usage_batch = call_groq_for_batch_fast(batch_page_infos)

        total_tokens += usage_batch.total_tokens
        input_tokens += usage_batch.input_tokens
        output_tokens += usage_batch.output_tokens

        # Map batch-local page_no → global page_no
        for i, page_dict in enumerate(raw_pages_batch):
            global_page_no = batch_start + i + 1  # 1-based for entire document
            page_dict["page_no"] = str(global_page_no)
            page_items = reconcile_page_items(page_dict)
            all_pages_fast.append(page_items)

    if not all_pages_fast:
        elapsed = time.time() - start_time
        print(
            f"[BILL_EXTRACT] pages={num_pages} items=0 total_amount=0.00 "
            f"tokens={total_tokens} time_sec={elapsed:.2f} (no items after fast pass)"
        )
        return ExtractBillDataResponse(
            is_success=False,
            token_usage=TokenUsage(
                total_tokens=total_tokens,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
            message="Model did not return any page items in fast pass.",
        )

    # 4. Find suspicious pages for high-accuracy recovery
    suspicious_idxs = find_suspicious_pages(all_pages_fast)

    # 5. HIGH-ACCURACY PASS (Maverick) for suspicious pages only
    all_pages_final: List[PageItems] = list(all_pages_fast)

    for idx in suspicious_idxs:
        # Safety check
        if idx < 0 or idx >= num_pages:
            continue

        high_page_dict, usage_hi = call_groq_single_page_accurate(page_infos[idx])

        total_tokens += usage_hi.total_tokens
        input_tokens += usage_hi.input_tokens
        output_tokens += usage_hi.output_tokens

        # Force correct global page_no
        high_page_dict["page_no"] = str(idx + 1)
        page_items_hi = reconcile_page_items(high_page_dict)

        all_pages_final[idx] = page_items_hi

    # 6. Aggregate
    data = aggregate_all_pages(all_pages_final)

    token_usage = TokenUsage(
        total_tokens=total_tokens,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    # 7. Logging
    grand_total = compute_grand_total_amount(all_pages_final)
    elapsed = time.time() - start_time

    print(
        f"[BILL_EXTRACT] pages={num_pages} suspicious={len(suspicious_idxs)} "
        f"items={data.total_item_count} total_amount={grand_total:.2f} "
        f"tokens={token_usage.total_tokens} time_sec={elapsed:.2f}"
    )

    return ExtractBillDataResponse(
        is_success=True,
        token_usage=token_usage,
        data=data,
    )
