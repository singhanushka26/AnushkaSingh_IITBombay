# ============================================================
#  main.py — Optimized, Stable, High-Accuracy Version (v11)
#  - Maverick Vision
#  - Stronger Prompt
#  - Unbreakable JSON Extractor
#  - Summary Filtering
#  - Deduplication
#  - Safe Numeric Enrichment
# ============================================================

import base64
import io
import json
import math
import mimetypes
import os
import time
from collections import defaultdict
from typing import List, Optional, Dict, Any

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from pdf2image import convert_from_bytes
from PIL import Image, ImageEnhance
from openai import OpenAI


# ============================================================
#  Groq Client
# ============================================================

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

# Maverick = highest accuracy
GROQ_VISION_MODEL_ID = "meta-llama/llama-4-maverick-17b-128e-instruct"

# max images per batch
MAX_IMAGES_PER_REQUEST = 5


# ============================================================
#  Schemas
# ============================================================

class BillItem(BaseModel):
    item_name: str
    item_amount: float
    item_rate: float
    item_quantity: float


class PageItems(BaseModel):
    page_no: str
    page_type: str
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
    token_usage: Optional[TokenUsage]
    data: Optional[ExtractBillDataResponseData] = None
    message: Optional[str] = None


# ============================================================
#  FastAPI App
# ============================================================

app = FastAPI(
    title="Bajaj Datathon API – Optimized Accuracy Engine",
    version="11.0",
)


# ============================================================
#  Download & Convert
# ============================================================

def download_document(url: str) -> bytes:
    try:
        r = requests.get(url, timeout=50)
        r.raise_for_status()
        return r.content
    except Exception as e:
        raise HTTPException(400, f"Failed to download document: {e}")


def guess_mime(url: str, content: bytes) -> str:
    mime, _ = mimetypes.guess_type(url)
    if mime:
        return mime
    if content.startswith(b"%PDF"):
        return "application/pdf"
    return "application/octet-stream"


def enhance_img(img: Image.Image):
    img = ImageEnhance.Contrast(img).enhance(1.35)
    img = ImageEnhance.Sharpness(img).enhance(1.25)
    return img


def resize_img(img: Image.Image, max_dim=1100):
    w, h = img.size
    scale = max(w, h) / max_dim
    if scale <= 1:
        return img
    return img.resize((int(w / scale), int(h / scale)), Image.LANCZOS)


def to_data_url(img: Image.Image):
    img = enhance_img(resize_img(img))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=66)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def document_to_pages(url: str, content: bytes):
    mime = guess_mime(url, content)
    pages = []

    if mime.startswith("image/"):
        img = Image.open(io.BytesIO(content)).convert("RGB")
        pages.append({"page_index": 0, "data_url": to_data_url(img)})
        return pages

    if mime == "application/pdf":
        pdf_pages = convert_from_bytes(content)
        for idx, p in enumerate(pdf_pages):
            img = p.convert("RGB")
            pages.append({"page_index": idx, "data_url": to_data_url(img)})
        return pages

    img = Image.open(io.BytesIO(content)).convert("RGB")
    pages.append({"page_index": 0, "data_url": to_data_url(img)})
    return pages


# ============================================================
#  STRONG UPGRADED PROMPT
# ============================================================

SYSTEM_PROMPT = """
You extract line items from medical bills with perfect structure.

EXTREMELY IMPORTANT RULES:

1. Each visible table ROW must map to EXACTLY ONE bill_item.
2. NEVER merge multiple rows into one.
3. NEVER break one row into multiple items.
4. NEVER hallucinate or invent items.
5. NEVER output totals, sub-totals, grand totals, round off, net amount, discounts.
6. item_name must match the bill EXACTLY (same spelling, spacing, punctuation).
7. item_quantity, item_rate, item_amount must match the table columns.
8. If numbers are unreadable, set them to 0.0 (DO NOT GUESS large values).
9. If multiple rows for same item have a single consistent numeric pattern,
   you MAY fill missing values using that pattern.
10. Output STRICT JSON ONLY. No markdown. No explanations. No ```json.

Output this schema exactly:
{
  "pagewise_line_items": [
    {
      "page_no": "1",
      "page_type": "Bill Detail" | "Pharmacy" | "Final Bill",
      "bill_items": [
        {
          "item_name": "...",
          "item_amount": 0.0,
          "item_rate": 0.0,
          "item_quantity": 0.0
        }
      ]
    }
  ],
  "total_item_count": 0
}
"""


# ============================================================
#  BULLETPROOF JSON EXTRACTOR
#  (Fixes truncated JSON, partial output, missing braces)
# ============================================================

def safe_json_extract(raw: str):
    # Remove markdown nonsense
    raw = raw.replace("```json", "").replace("```", "").strip()

    # Find JSON boundaries
    first = raw.find("{")
    last = raw.rfind("}")

    if first == -1:
        raise HTTPException(500, f"Model returned no JSON: {raw[:200]}")

    if last == -1 or last < first:
        text = raw[first:]
    else:
        text = raw[first:last+1]

    # Basic cleanup
    text = text.replace(",}", "}").replace(",]", "]")

    # Brace repair
    need_curly = text.count("{") - text.count("}")
    if need_curly > 0:
        text += "}" * need_curly

    need_square = text.count("[") - text.count("]")
    if need_square > 0:
        text += "]" * need_square

    # Try parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try trimming trailing garbage
        for cut in range(len(text)-1, 0, -1):
            try:
                return json.loads(text[:cut])
            except:
                continue

    raise HTTPException(500, f"Could not recover JSON: {raw[:200]}")


# ============================================================
#  LLM BATCH CALL
# ============================================================

def call_groq(batch):
    user_content = [{"type": "input_text", "text": f"{len(batch)} pages"}]

    for i, pg in enumerate(batch):
        user_content.append({"type": "input_text", "text": f"PAGE {i+1}:"})
        user_content.append({
            "type": "input_image",
            "image_url": pg["data_url"],
            "detail": "high"
        })

    response = client.responses.create(
        model=GROQ_VISION_MODEL_ID,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": user_content},
        ],
    )

    usage = response.usage or {}
    usage_tok = TokenUsage(
        total_tokens=int(getattr(usage, "total_tokens", 0)),
        input_tokens=int(getattr(usage, "input_tokens", 0)),
        output_tokens=int(getattr(usage, "output_tokens", 0)),
    )

    parsed = safe_json_extract(response.output_text)
    raw_pages = parsed.get("pagewise_line_items", [])

    return raw_pages, usage_tok


# ============================================================
#  Cleaning, Summary Removal, Dedup
# ============================================================

SUMMARY_WORDS = ("TOTAL", "SUB", "NET", "GRAND", "ROUND", "BALANCE", "DISCOUNT")


def clean_page_dict(pg):
    out = []

    for item in pg.get("bill_items", []):
        name = str(item.get("item_name", "")).strip()

        if any(w in name.upper() for w in SUMMARY_WORDS):
            continue

        out.append({
            "item_name": name,
            "item_amount": float(item.get("item_amount", 0)),
            "item_rate": float(item.get("item_rate", 0)),
            "item_quantity": float(item.get("item_quantity", 0)),
        })

    pg["bill_items"] = out
    return pg


def dedupe_page(pg: PageItems):
    seen = set()
    final = []

    for it in pg.bill_items:
        key = (it.item_name.lower(), it.item_amount, it.item_rate, it.item_quantity)
        if key in seen:
            continue
        seen.add(key)
        final.append(it)

    pg.bill_items = final
    return pg


# ============================================================
#  Safe Numeric Enrichment
# ============================================================

def enrich_from_patterns(pages):
    patterns = defaultdict(set)

    for pg in pages:
        for it in pg.bill_items:
            if it.item_rate > 0 and it.item_quantity > 0:
                patterns[it.item_name.lower()].add((it.item_rate, it.item_quantity))

    defaults = {k: list(v)[0] for k, v in patterns.items() if len(v) == 1}

    for pg in pages:
        for it in pg.bill_items:
            key = it.item_name.lower()
            if key not in defaults:
                continue

            dr, dq = defaults[key]

            if it.item_quantity == 0:
                it.item_quantity = dq
            if it.item_rate == 0:
                it.item_rate = dr
            if it.item_amount == 0:
                it.item_amount = it.item_rate * it.item_quantity

            it.item_rate = round(it.item_rate, 2)
            it.item_amount = round(it.item_amount, 2)

    return pages


# ============================================================
#  Aggregate
# ============================================================

def aggregate_pages(pages):
    return ExtractBillDataResponseData(
        pagewise_line_items=pages,
        total_item_count=sum(len(p.bill_items) for p in pages)
    )


# ============================================================
#  ENDPOINTS
# ============================================================

@app.get("/extract-bill-data")
def health():
    return {"message": "Health OK"}


@app.post("/extract-bill-data", response_model=ExtractBillDataResponse)
def extract(req: ExtractBillDataRequest):
    start = time.time()

    content = download_document(str(req.document))
    pages_raw = document_to_pages(str(req.document), content)

    all_pages = []
    usage_total = TokenUsage(total_tokens=0, input_tokens=0, output_tokens=0)

    for i in range(0, len(pages_raw), MAX_IMAGES_PER_REQUEST):
        batch = pages_raw[i:i+MAX_IMAGES_PER_REQUEST]
        raw_pages, usage = call_groq(batch)

        usage_total.total_tokens += usage.total_tokens
        usage_total.input_tokens += usage.input_tokens
        usage_total.output_tokens += usage.output_tokens

        for j, pg in enumerate(raw_pages):
            pg["page_no"] = str(i + j + 1)
            pg = clean_page_dict(pg)

            page_obj = PageItems(**pg)
            page_obj = dedupe_page(page_obj)

            all_pages.append(page_obj)

    all_pages = enrich_from_patterns(all_pages)
    data = aggregate_pages(all_pages)

    elapsed = time.time() - start
    print(f"[EXTRACT] pages={len(all_pages)} items={data.total_item_count} tokens={usage_total.total_tokens} time={elapsed:.2f}s")

    return ExtractBillDataResponse(
        is_success=True,
        token_usage=usage_total,
        data=data
    )
