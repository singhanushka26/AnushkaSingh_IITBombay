import base64
import io
import json
import math
import mimetypes
import os
import time
from typing import List, Optional, Tuple, Any, Dict, Set

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from pdf2image import convert_from_bytes
from PIL import Image
from openai import OpenAI

# ============================================================
#  Groq Client
# ============================================================

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

MODEL_FAST = "meta-llama/llama-4-scout-17b-16e-instruct"
MODEL_ACCURATE = "meta-llama/llama-4-maverick-17b-128e-instruct"


# ============================================================
#  Schemas (STRICT REQUIREMENTS FROM DATATHON)
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

app = FastAPI()


# ============================================================
#  Helpers — Download + Convert
# ============================================================

def download_document(url: str) -> bytes:
    try:
        r = requests.get(url, timeout=40)
        r.raise_for_status()
        return r.content
    except Exception as e:
        raise HTTPException(400, f"Failed to download document: {e}")


def guess_mime(url: str, content: bytes) -> str:
    mime, _ = mimetypes.guess_type(url)
    if mime:
        return mime
    if content[:4] == b"%PDF":
        return "application/pdf"
    return "application/octet-stream"


def resize_for_llm(img: Image.Image) -> Image.Image:
    max_dim = 1100
    w, h = img.size
    s = max(w, h) / max_dim
    if s <= 1:
        return img
    return img.resize((int(w/s), int(h/s)), Image.LANCZOS)


def img_to_data_url(img: Image.Image) -> str:
    img = resize_for_llm(img)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=60)
    data = buf.getvalue()

    while len(data) > 4*1024*1024:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=40)
        data = buf.getvalue()

    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def document_to_images(url: str, content: bytes) -> List[str]:
    mime = guess_mime(url, content)
    pages = []

    if mime.startswith("image/"):
        img = Image.open(io.BytesIO(content)).convert("RGB")
        pages.append(img_to_data_url(img))
        return pages

    if mime == "application/pdf":
        pdf_pages = convert_from_bytes(content)
        for p in pdf_pages:
            pages.append(img_to_data_url(p.convert("RGB")))
        return pages

    # fallback
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
        pages.append(img_to_data_url(img))
        return pages
    except:
        raise HTTPException(400, f"Unsupported file format: {mime}")


# ============================================================
#  PROMPTS — Optimized for JSON ONLY
# ============================================================

SYSTEM_PROMPT = """
You extract line items from MULTI-PAGE hospital bills.

RULES:
- Output STRICT JSON ONLY.
- NO markdown, NO ```json, NO headings, NO explanations.
- NEVER output totals, sub-totals, tax rows, headers.
- NEVER duplicate the same item across pages.
- item_amount must match the row amount printed in the bill.
- If qty missing → qty=1 and rate=item_amount.
- If rate missing → rate=item_amount/qty.
- Numeric values must be JSON numbers.
- page_no as STRING, starting from "1".
"""

USER_PROMPT_GLOBAL = """
You will receive several page images in order.
For EACH page:
- Identify page_type ∈ {"Bill Detail", "Final Bill", "Pharmacy"}.
- Extract bill_items ONLY from that page.
- Do NOT copy summary rows from Final Bill pages.
- Do NOT double count items also present in detail pages.

Output ONE strict JSON object:
{
  "pagewise_line_items": [
    { "page_no": "1", "page_type": "...", "bill_items": [...] },
    { "page_no": "2", "page_type": "...", "bill_items": [...] }
  ],
  "total_item_count": <int>
}
"""


USER_PROMPT_SINGLE = """
You are re-checking ONLY ONE PAGE (Final Bill page).
Extract ONLY real item rows.
Do NOT output totals or repeated summary rows.
Output STRICT JSON ONLY for this page.
"""


# ============================================================
#  Groq Call Helpers
# ============================================================

def groq_call(model: str, system: str, user_blocks: List[dict]) -> Tuple[dict, TokenUsage]:
    try:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": system}]},
                {"role": "user", "content": user_blocks},
            ],
        )
    except Exception as e:
        raise HTTPException(503, f"Groq API error: {e}")

    usage = getattr(resp, "usage", None)
    tu = TokenUsage(
        total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
    )

    raw = resp.output_text.strip()

    # Strip markdown
    if raw.startswith("```"):
        raw = raw.strip("`")

    # Extract pure JSON
    try:
        return json.loads(raw), tu
    except:
        first = raw.find("{")
        last = raw.rfind("}")
        if first != -1 and last != -1:
            try:
                return json.loads(raw[first:last+1]), tu
            except:
                raise HTTPException(500, f"Invalid JSON: {raw[:200]}")
        raise HTTPException(500, f"Invalid JSON: {raw[:200]}")


# ============================================================
#  Cleaning
# ============================================================

def norm_num(x):
    if x is None:
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.replace(",", "").strip()
        if s in ["", "-", "—"]:
            return 0.0
        try:
            return float(s)
        except:
            return 0.0
    return 0.0


def clean_page(p: dict) -> dict:
    items = []
    seen = set()

    for it in p.get("bill_items", []) or []:
        name = str(it.get("item_name", "")).strip()
        amt = norm_num(it.get("item_amount"))
        rate = norm_num(it.get("item_rate"))
        qty = norm_num(it.get("item_quantity"))

        key = (name.lower(), round(amt, 2), round(rate, 4), round(qty, 4))
        if key in seen:
            continue
        seen.add(key)

        items.append({
            "item_name": name,
            "item_amount": amt,
            "item_rate": rate,
            "item_quantity": qty
        })

    return {
        "page_no": p.get("page_no", ""),
        "page_type": p.get("page_type", "Bill Detail"),
        "bill_items": items,
    }


def fix_math(page: PageItems):
    EPS = 0.02
    for it in page.bill_items:
        amt = float(it.item_amount)
        rate = float(it.item_rate)
        qty = float(it.item_quantity)

        if rate > 0 and qty > 0:
            if abs(rate * qty - amt) > EPS:
                it.item_amount = round(rate * qty, 2)
        elif qty > 0 and amt > 0:
            it.item_rate = round(amt / qty, 4)
        elif amt > 0:
            it.item_quantity = 1.0
            it.item_rate = round(amt, 2)
    return page


def dedupe(pages: List[PageItems]):
    seen = set()
    out = []
    for p in pages:
        new_items = []
        for it in p.bill_items:
            key = (
                p.page_type.lower(),
                it.item_name.lower(),
                round(float(it.item_rate), 4),
                round(float(it.item_quantity), 4),
                round(float(it.item_amount), 2),
            )
            if key in seen:
                continue
            seen.add(key)
            new_items.append(it)
        p.bill_items = new_items
        out.append(p)
    return out


# ============================================================
#  GET Health
# ============================================================

@app.get("/extract-bill-data")
def health():
    return {"status": "OK"}


# ============================================================
#  POST Endpoint
# ============================================================

@app.post("/extract-bill-data", response_model=ExtractBillDataResponse)
def extract_api(req: ExtractBillDataRequest):
    t0 = time.time()

    content = download_document(str(req.document))
    images = document_to_images(str(req.document), content)
    num_pages = len(images)

    # ----------------------------
    # 1) FAST BATCH PASS (Scout)
    # ----------------------------
    batches = []
    for i in range(0, num_pages, 5):
        block = [{"type": "input_text", "text": f"Pages {i+1}-{min(i+5,num_pages)}:"}]
        for j in range(i, min(i+5, num_pages)):
            block.append({"type": "input_text", "text": f"PAGE {j+1}:"})
            block.append({"type": "input_image", "image_url": images[j], "detail": "auto"})
        batches.append(block)

    merged_pages = []
    total_usage = TokenUsage(0, 0, 0)

    for b in batches:
        parsed, u = groq_call(MODEL_FAST, SYSTEM_PROMPT, [{"type": "input_text", "text": USER_PROMPT_GLOBAL}] + b)
        total_usage.total_tokens += u.total_tokens
        total_usage.input_tokens += u.input_tokens
        total_usage.output_tokens += u.output_tokens

        pages = parsed.get("pagewise_line_items", parsed if isinstance(parsed, list) else [])
        merged_pages += pages

    # Sort & Normalize
    pages_dict = {}
    for idx in range(num_pages):
        pages_dict[idx] = {"page_no": str(idx+1), "page_type": "Bill Detail", "bill_items": []}

    for p in merged_pages:
        idx = int(p.get("page_no", "1")) - 1
        if idx in pages_dict:
            pages_dict[idx] = p

    # ----------------------------
    # 2) ACCURACY PASS (Maverick)
    # ----------------------------
    for idx, page in pages_dict.items():
        if page.get("page_type", "").lower().strip() == "final bill":
            parsed, u = groq_call(
                MODEL_ACCURATE,
                SYSTEM_PROMPT,
                [
                    {"type": "input_text", "text": USER_PROMPT_SINGLE},
                    {"type": "input_image", "image_url": images[idx], "detail": "high"}
                ],
            )
            total_usage.total_tokens += u.total_tokens
            total_usage.input_tokens += u.input_tokens
            total_usage.output_tokens += u.output_tokens

            pages_dict[idx] = parsed

    # ----------------------------
    # 3) CLEANING
    # ----------------------------
    cleaned = []
    for i in range(num_pages):
        p = clean_page(pages_dict[i])
        page = PageItems(**p)
        page = fix_math(page)
        cleaned.append(page)

    cleaned = dedupe(cleaned)

    # Final aggregation
    total_items = sum(len(p.bill_items) for p in cleaned)
    data = ExtractBillDataResponseData(
        pagewise_line_items=cleaned,
        total_item_count=total_items
    )

    print(f"[BILL] pages={num_pages} items={total_items} tokens={total_usage.total_tokens} time={time.time()-t0:.2f}s")

    return ExtractBillDataResponse(
        is_success=True,
        token_usage=total_usage,
        data=data
    )
