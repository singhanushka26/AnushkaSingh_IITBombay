# ============================
#   main.py  (High Accuracy)
#   OCR + Maverick + JSON Fix
#   Version 7.2
# ============================

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
import pytesseract

# ============================================================
#  Groq Client (OpenAI-compatible)
# ============================================================

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

# High accuracy model
GROQ_VISION_MODEL_ID = os.environ.get(
    "GROQ_VISION_MODEL_ID",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
)

MAX_IMAGES_PER_REQUEST = 5  # Groq Vision limit


# ============================================================
#  Schemas (Exact HackRx Format)
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
    token_usage: Optional[TokenUsage] = None
    data: Optional[ExtractBillDataResponseData] = None
    message: Optional[str] = None


# ============================================================
#  FastAPI App
# ============================================================

app = FastAPI(
    title="Bajaj Datathon API",
    version="7.2-high-accuracy",
    description="OCR + Maverick + robust JSON parsing",
)


# ============================================================
#  Download & Decode Document
# ============================================================

def download_document(url: str) -> bytes:
    try:
        resp = requests.get(url, timeout=40)
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        raise HTTPException(400, f"Download failed: {e}")


def guess_mime_type(url: str, content: bytes) -> str:
    mime, _ = mimetypes.guess_type(url)
    if mime:
        return mime
    if content[:4] == b"%PDF":
        return "application/pdf"
    return "application/octet-stream"


# ============================================================
#  Image Preprocessing + OCR
# ============================================================

def _resize_for_vision(img: Image.Image, max_dim: int = 950) -> Image.Image:
    w, h = img.size
    scale = max(w, h) / float(max_dim)
    if scale <= 1:
        return img
    return img.resize((int(w/scale), int(h/scale)), Image.LANCZOS)


def _enhance_for_vision(img: Image.Image) -> Image.Image:
    img = ImageEnhance.Contrast(img).enhance(1.6)
    img = ImageEnhance.Sharpness(img).enhance(1.4)
    return img


def _ocr_page(img: Image.Image) -> str:
    try:
        gray = img.convert("L")
        gray = ImageEnhance.Contrast(gray).enhance(1.8)
        return pytesseract.image_to_string(gray, config="--psm 6")
    except Exception:
        return ""


def image_to_data_url(img: Image.Image, quality: int = 60) -> str:
    img = _resize_for_vision(img)
    img = _enhance_for_vision(img)

    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    b = buf.getvalue()

    while len(b) > 4*1024*1024 and quality > 30:
        quality -= 10
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=quality)
        b = buf.getvalue()

    return f"data:image/jpeg;base64,{base64.b64encode(b).decode()}"


def document_to_page_infos(url: str, content: bytes) -> List[Dict[str, Any]]:
    mime = guess_mime_type(url, content)
    pages = []

    if mime.startswith("image/"):
        img = Image.open(io.BytesIO(content)).convert("RGB")
        pages.append({
            "page_index": 0,
            "data_url": image_to_data_url(img),
            "ocr_text": _ocr_page(img),
        })
        return pages

    if mime == "application/pdf":
        pdf_imgs = convert_from_bytes(content)
        for idx, p in enumerate(pdf_imgs):
            img = p.convert("RGB")
            pages.append({
                "page_index": idx,
                "data_url": image_to_data_url(img),
                "ocr_text": _ocr_page(img),
            })
        return pages

    img = Image.open(io.BytesIO(content)).convert("RGB")
    pages.append({
        "page_index": 0,
        "data_url": image_to_data_url(img),
        "ocr_text": _ocr_page(img),
    })
    return pages


# ============================================================
#  LLM Prompt (OCR + Image Hybrid)
# ============================================================

SYSTEM_PROMPT = """
You extract ALL line items from hospital bills with PERFECT recall.
You receive OCR TEXT + IMAGE for each page.

Rules:
- Use OCR TEXT as PRIMARY source of table rows.
- For EVERY row describing a charge, output EXACTLY ONE bill item.
- DO NOT merge rows.
- DO NOT hallucinate new rows.
- DO NOT output totals, sub-totals, GRAND TOTAL, NET AMOUNT, ROUND OFF, DISCOUNT.
- Preserve exact item names.
- item_amount must be the NET amount for that row.
- If numbers missing but same item appears with consistent pattern, infer safely.
- Otherwise leave missing numeric fields as 0.0.

Output STRICT JSON ONLY:
{
  "pagewise_line_items": [
    { "page_no": "1",
      "page_type": "Bill Detail" | "Final Bill" | "Pharmacy",
      "bill_items": [
        {"item_name": "...", "item_amount": 0.0, "item_rate": 0.0, "item_quantity": 0.0}
      ]
    }
  ],
  "total_item_count": 0
}
"""

# ============================================================
#  JSON Extractor Patch  (Bulletproof)
# ============================================================

def extract_json_safe(raw: str) -> Any:
    """Safely extract JSON from model output."""
    raw = raw.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    first = raw.find("{")
    last = raw.rfind("}")
    if first == -1 or last == -1 or last <= first:
        raise HTTPException(500, f"Model returned invalid JSON: {raw[:200]}")

    core = raw[first:last+1]

    try:
        return json.loads(core)
    except Exception:
        cleaned = core.replace(",}", "}").replace(",]", "]")
        try:
            return json.loads(cleaned)
        except Exception:
            raise HTTPException(500, f"Model JSON parse failed: {raw[:200]}")


# ============================================================
#  Groq Batch Call
# ============================================================

def call_groq_for_batch(batch_pages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], TokenUsage]:
    num = len(batch_pages)

    user_parts = [{
        "type": "input_text",
        "text": f"You are given {num} pages. Use OCR TEXT as source of rows.",
    }]

    for i, pg in enumerate(batch_pages):
        idx = i + 1
        ocr = pg["ocr_text"].strip()

        user_parts.append({"type": "input_text", "text": f"--- OCR PAGE {idx} ---\n{ocr}"})
        user_parts.append({"type": "input_image", "image_url": pg["data_url"], "detail": "auto"})

    resp = client.responses.create(
        model=GROQ_VISION_MODEL_ID,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": user_parts},
        ],
    )

    usage = getattr(resp, "usage", None)
    tokens = TokenUsage(
        total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
    )

    raw_output = resp.output_text
    parsed = extract_json_safe(raw_output)

    raw_pages = parsed.get("pagewise_line_items", [])

    if len(raw_pages) < num:
        for k in range(len(raw_pages), num):
            raw_pages.append({
                "page_no": str(k+1),
                "page_type": "Bill Detail",
                "bill_items": []
            })

    return raw_pages, tokens


# ============================================================
#  Cleaning, Deduplication, & Enrichment
# ============================================================

SUMMARY_WORDS = (
    "TOTAL", "SUB TOTAL", "SUB-TOTAL", "GRAND TOTAL",
    "NET AMOUNT", "NET PAYABLE", "ROUND OFF", "DISCOUNT"
)

def _num(x):  # safe numeric coercion
    if not x:
        return 0.0
    try:
        return float(str(x).replace(",", "").strip())
    except:
        return 0.0


def clean_page_dict(pg: Dict[str, Any]) -> Dict[str, Any]:
    out = []
    for item in pg.get("bill_items", []):
        name = str(item.get("item_name", "")).strip()
        if not name:
            continue

        if any(w in name.upper() for w in SUMMARY_WORDS):
            continue

        out.append({
            "item_name": name,
            "item_amount": _num(item.get("item_amount")),
            "item_rate": _num(item.get("item_rate")),
            "item_quantity": _num(item.get("item_quantity")),
        })

    return {
        "page_no": str(pg.get("page_no", "")),
        "page_type": str(pg.get("page_type", "Bill Detail")),
        "bill_items": out,
    }


def reconcile_page_items(pg: Dict[str, Any]) -> PageItems:
    cleaned = clean_page_dict(pg)
    try:
        page = PageItems(**cleaned)
    except Exception as e:
        raise HTTPException(500, f"Schema mismatch: {e}")

    EPS = 0.01
    for it in page.bill_items:
        amt, rate, qty = it.item_amount, it.item_rate, it.item_quantity

        if rate > 0 and qty > 0:
            comp = rate * qty
            if abs(comp - amt) > EPS:
                it.item_amount = round(comp, 2)
        elif amt > 0 and qty > 0:
            it.item_rate = round(amt / qty, 4)
        elif amt > 0 and qty == 0:
            it.item_quantity = 1.0
            it.item_rate = amt

    return page


def dedupe_with_ocr(page: PageItems, ocr_text: str) -> PageItems:
    txt = (ocr_text or "").lower()
    name_counts = defaultdict(int)
    for i in page.bill_items:
        name_counts[i.item_name.lower()] += 1

    allowed = {}
    for nm, cnt in name_counts.items():
        occ = txt.count(nm)
        allowed[nm] = max(1, min(cnt, occ if occ > 0 else cnt))

    seen = defaultdict(int)
    new = []
    for it in page.bill_items:
        nm = it.item_name.lower()
        if seen[nm] < allowed[nm]:
            new.append(it)
            seen[nm] += 1

    page.bill_items = new
    return page


def enrich_from_patterns(pages: List[PageItems]):
    patterns = defaultdict(set)

    for pg in pages:
        for it in pg.bill_items:
            if it.item_rate > 0 and it.item_quantity > 0:
                patterns[it.item_name.lower()].add((round(it.item_rate, 4), round(it.item_quantity, 4)))
            elif it.item_amount > 0 and it.item_quantity > 0:
                patterns[it.item_name.lower()].add((round(it.item_amount/it.item_quantity, 4),
                                                    round(it.item_quantity, 4)))

    defaults = {}
    for k, s in patterns.items():
        if len(s) == 1:
            defaults[k] = list(s)[0]

    for pg in pages:
        for it in pg.bill_items:
            key = it.item_name.lower()
            if key not in defaults:
                continue

            dr, dq = defaults[key]

            if it.item_quantity <= 0:
                it.item_quantity = dq
            if it.item_rate <= 0 and it.item_amount > 0:
                it.item_rate = it.item_amount / it.item_quantity
            if it.item_rate <= 0:
                it.item_rate = dr
            if it.item_amount <= 0:
                it.item_amount = it.item_rate * it.item_quantity

            it.item_rate = round(it.item_rate, 2)
            it.item_amount = round(it.item_amount, 2)

    return pages


# ============================================================
#  Aggregation
# ============================================================

def aggregate_all(pages: List[PageItems]) -> ExtractBillDataResponseData:
    total = sum(len(p.bill_items) for p in pages)
    return ExtractBillDataResponseData(
        pagewise_line_items=pages,
        total_item_count=total
    )


# ============================================================
#  API Endpoints
# ============================================================

@app.get("/extract-bill-data")
def health():
    return {
        "message": "Health OK. Use POST /extract-bill-data with JSON body {\"document\": \"<URL>\"}"
    }


@app.post("/extract-bill-data", response_model=ExtractBillDataResponse)
def extract(req: ExtractBillDataRequest):
    start = time.time()
    url = str(req.document)

    content = download_document(url)
    pages_info = document_to_page_infos(url, content)

    all_pages = []
    usage_total = TokenUsage(0, 0, 0)

    for i in range(0, len(pages_info), MAX_IMAGES_PER_REQUEST):
        batch = pages_info[i:i+MAX_IMAGES_PER_REQUEST]

        raw_pages, usage = call_groq_for_batch(batch)
        usage_total.total_tokens += usage.total_tokens
        usage_total.input_tokens += usage.input_tokens
        usage_total.output_tokens += usage.output_tokens

        for j, pgdict in enumerate(raw_pages):
            pgdict["page_no"] = str(i + j + 1)

            page = reconcile_page_items(pgdict)
            page = dedupe_with_ocr(page, batch[j]["ocr_text"])
            all_pages.append(page)

    all_pages = enrich_from_patterns(all_pages)
    data = aggregate_all(all_pages)

    elapsed = time.time() - start
    print(f"[BILL] pages={len(all_pages)} items={data.total_item_count} tokens={usage_total.total_tokens} time={elapsed:.2f}s")

    return ExtractBillDataResponse(
        is_success=True,
        token_usage=usage_total,
        data=data
    )
