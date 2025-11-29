# ============================================================
#  main.py — Bajaj Datathon High Accuracy Pipeline (Option B)
#  OCR + Maverick Vision + Strong Prompt + Dedup + JSON Repair
#  Version: 9.0 (Stable)
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
import pytesseract

# ============================================================
#  Groq Client (OpenAI-compatible)
# ============================================================

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

# Use Maverick (highest accuracy)
GROQ_VISION_MODEL_ID = "meta-llama/llama-4-maverick-17b-128e-instruct"
MAX_IMAGES_PER_REQUEST = 5


# ============================================================
#  Response Schemas (Exact HackRx Format)
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
    data: Optional[ExtractBillDataResponseData]
    message: Optional[str] = None


# ============================================================
#  FastAPI Application
# ============================================================

app = FastAPI(
    title="Bajaj Datathon – High Accuracy Extraction Engine",
    version="9.0",
    description="OCR + Maverick + Hybrid Table Extraction",
)


# ============================================================
#  Download Utility
# ============================================================

def download_document(url: str) -> bytes:
    try:
        r = requests.get(url, timeout=40)
        r.raise_for_status()
        return r.content
    except Exception as e:
        raise HTTPException(400, f"Failed to download: {e}")


def guess_mime(url: str, content: bytes) -> str:
    mime, _ = mimetypes.guess_type(url)
    if mime:
        return mime
    if content.startswith(b"%PDF"):
        return "application/pdf"
    return "application/octet-stream"


# ============================================================
#  Image Processing + OCR
# ============================================================

def _enhance(img: Image.Image) -> Image.Image:
    img = ImageEnhance.Contrast(img).enhance(1.6)
    img = ImageEnhance.Sharpness(img).enhance(1.4)
    return img


def _resize(img: Image.Image, max_dim: int = 950) -> Image.Image:
    w, h = img.size
    scale = max(w, h) / max_dim
    if scale <= 1:
        return img
    return img.resize((int(w/scale), int(h/scale)), Image.LANCZOS)


def to_data_url(img: Image.Image) -> str:
    img = _resize(_enhance(img))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=60)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def ocr_page(img: Image.Image) -> str:
    try:
        gray = img.convert("L")
        gray = ImageEnhance.Contrast(gray).enhance(1.8)
        txt = pytesseract.image_to_string(gray, config="--psm 6")
        return txt
    except Exception:
        return ""


def load_pages(url: str, content: bytes):
    mime = guess_mime(url, content)
    out = []

    if mime.startswith("image/"):
        img = Image.open(io.BytesIO(content)).convert("RGB")
        out.append({
            "page_index": 0,
            "ocr": ocr_page(img),
            "img": to_data_url(img)
        })
        return out

    if mime == "application/pdf":
        pages = convert_from_bytes(content)
        for i, p in enumerate(pages):
            img = p.convert("RGB")
            out.append({
                "page_index": i,
                "ocr": ocr_page(img),
                "img": to_data_url(img)
            })
        return out

    img = Image.open(io.BytesIO(content)).convert("RGB")
    out.append({
        "page_index": 0,
        "ocr": ocr_page(img),
        "img": to_data_url(img)
    })
    return out


# ============================================================
#  LLM Prompt
# ============================================================

SYSTEM_PROMPT = """
You extract ALL line items from hospital bills with PERFECT recall.

You are given OCR TEXT + IMAGE for each page.

RULES:
- OCR TEXT is the PRIMARY truth for all row boundaries.
- For EVERY valid charge row in OCR, output EXACTLY one bill item.
- NEVER merge two OCR rows.
- NEVER hallucinate items not in OCR.
- NEVER output: TOTAL, SUB TOTAL, GRAND TOTAL, NET AMOUNT, DISCOUNT, ROUND OFF.
- Use IMAGE ONLY for confirmation, NOT for row count.

NUMERIC RULES:
- item_amount is NET AMOUNT for that row.
- If row has amount missing but appears multiple times with consistent pattern, infer safely.
- Otherwise leave missing values = 0.0.

OUTPUT STRICT JSON ONLY (no markdown).
"""


# ============================================================
#  Bulletproof JSON Extractor
# ============================================================

def safe_json_extract(raw: str) -> Any:
    raw = raw.replace("```json", "").replace("```", "").strip()
    first = raw.find("{")
    last = raw.rfind("}")

    if first == -1 or last == -1 or last <= first:
        raise HTTPException(500, "Invalid JSON from model")

    core = raw[first:last+1]

    try:
        return json.loads(core)
    except:
        repaired = core.replace(",}", "}").replace(",]", "]")
        return json.loads(repaired)


# ============================================================
#  Groq Batch Call
# ============================================================

def call_batch(batch):
    parts = [{"type": "input_text", "text": f"{len(batch)} pages follow"}]

    for i, pg in enumerate(batch):
        parts.append({"type": "input_text", "text": f"OCR PAGE {i+1}:\n{pg['ocr']}"})
        parts.append({"type": "input_image", "image_url": pg['img'], "detail": "auto"})

    resp = client.responses.create(
        model=GROQ_VISION_MODEL_ID,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": parts},
        ],
    )

    usage = resp.usage or {}
    safe = TokenUsage(
        total_tokens=int(getattr(usage, "total_tokens", 0)),
        input_tokens=int(getattr(usage, "input_tokens", 0)),
        output_tokens=int(getattr(usage, "output_tokens", 0)),
    )

    parsed = safe_json_extract(resp.output_text)
    pages = parsed.get("pagewise_line_items", [])
    return pages, safe


# ============================================================
#  Cleaning, Summary Removal, Dedup
# ============================================================

SUMMARY_WORDS = ("TOTAL", "SUB TOTAL", "NET", "GRAND", "ROUND", "DISCOUNT")


def clean_page(pg):
    out = []
    for item in pg["bill_items"]:
        name = str(item.get("item_name", "")).strip()
        if any(w in name.upper() for w in SUMMARY_WORDS):
            continue

        out.append({
            "item_name": name,
            "item_amount": float(str(item.get("item_amount", 0)).replace(",", "") or 0),
            "item_rate": float(str(item.get("item_rate", 0)).replace(",", "") or 0),
            "item_quantity": float(str(item.get("item_quantity", 0)).replace(",", "") or 0),
        })

    pg["bill_items"] = out
    return pg


def dedupe_with_ocr(page_items: PageItems, ocr: str):
    txt = (ocr or "").lower()
    seen = defaultdict(int)
    allowed = defaultdict(int)

    for it in page_items.bill_items:
        nm = it.item_name.lower()
        allowed[nm] = txt.count(nm) if txt.count(nm) > 0 else len(page_items.bill_items)

    final = []
    for it in page_items.bill_items:
        nm = it.item_name.lower()
        if seen[nm] < allowed[nm]:
            final.append(it)
            seen[nm] += 1

    page_items.bill_items = final
    return page_items


# ============================================================
#  Numeric Enrichment
# ============================================================

def enrich_patterns(pages):
    patterns = defaultdict(set)

    for pg in pages:
        for it in pg.bill_items:
            if it.item_rate > 0 and it.item_quantity > 0:
                patterns[it.item_name.lower()].add((it.item_rate, it.item_quantity))
            elif it.item_amount > 0 and it.item_quantity > 0:
                patterns[it.item_name.lower()].add((it.item_amount/it.item_quantity, it.item_quantity))

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

def aggregate(pages):
    return ExtractBillDataResponseData(
        pagewise_line_items=pages,
        total_item_count=sum(len(p.bill_items) for p in pages)
    )


# ============================================================
#  Health
# ============================================================

@app.get("/extract-bill-data")
def health():
    return {"message": "Health OK"}


# ============================================================
#  Main Endpoint
# ============================================================

@app.post("/extract-bill-data", response_model=ExtractBillDataResponse)
def extract(req: ExtractBillDataRequest):
    start = time.time()
    content = download_document(str(req.document))
    pages = load_pages(str(req.document), content)

    all_pages = []
    usage_total = TokenUsage(total_tokens=0, input_tokens=0, output_tokens=0)

    for i in range(0, len(pages), MAX_IMAGES_PER_REQUEST):
        batch = pages[i:i+MAX_IMAGES_PER_REQUEST]

        raw_pages, usage = call_batch(batch)
        usage_total.total_tokens += usage.total_tokens
        usage_total.input_tokens += usage.input_tokens
        usage_total.output_tokens += usage.output_tokens

        for j, pg in enumerate(raw_pages):
            pg_clean = clean_page(pg)
            pg_clean["page_no"] = str(i+j+1)

            page_obj = PageItems(**pg_clean)
            page_obj = dedupe_with_ocr(page_obj, batch[j]['ocr'])
            all_pages.append(page_obj)

    all_pages = enrich_patterns(all_pages)
    data = aggregate(all_pages)

    print(f"[BILL] pages={len(all_pages)} items={data.total_item_count} tokens={usage_total.total_tokens} time={time.time()-start:.2f}s")

    return ExtractBillDataResponse(
        is_success=True,
        token_usage=usage_total,
        data=data
    )
