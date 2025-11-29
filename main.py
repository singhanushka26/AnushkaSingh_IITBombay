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

# Fast + cheap
MODEL_FAST = "meta-llama/llama-4-scout-17b-16e-instruct"
# More accurate, used only on Final Bill pages
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

app = FastAPI(
    title="Bajaj Datathon Bill Extraction API",
    version="5.0.0",
)


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
    # Smaller max_dim → fewer tokens → faster
    max_dim = 1100
    w, h = img.size
    s = max(w, h) / max_dim
    if s <= 1:
        return img
    return img.resize((int(w / s), int(h / s)), Image.LANCZOS)


def img_to_data_url(img: Image.Image) -> str:
    img = resize_for_llm(img)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=60)
    data = buf.getvalue()

    # Safety: keep below ~4MB per image for base64
    while len(data) > 4 * 1024 * 1024:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=40)
        data = buf.getvalue()

    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def document_to_images(url: str, content: bytes) -> List[str]:
    mime = guess_mime(url, content)
    pages: List[str] = []

    if mime.startswith("image/"):
        img = Image.open(io.BytesIO(content)).convert("RGB")
        pages.append(img_to_data_url(img))
        return pages

    if mime == "application/pdf":
        try:
            pdf_pages = convert_from_bytes(content)
        except Exception as e:
            raise HTTPException(
                400,
                f"Unable to convert PDF to images. Ensure poppler is installed. Error: {e}",
            )
        for p in pdf_pages:
            pages.append(img_to_data_url(p.convert("RGB")))
        return pages

    # fallback as image
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
        pages.append(img_to_data_url(img))
        return pages
    except Exception:
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
You are re-checking ONLY ONE PAGE (usually a Final Bill page).
Extract ONLY real item rows on THIS PAGE.
Do NOT output totals, taxes, discounts, or repeated summary rows.
Output STRICT JSON ONLY for this page, with shape:
{
  "page_no": "<string>",
  "page_type": "Bill Detail" | "Final Bill" | "Pharmacy",
  "bill_items": [
    { "item_name": "...", "item_amount": <float>, "item_rate": <float>, "item_quantity": <float> }
  ]
}
"""


# ============================================================
#  Groq Call Helpers
# ============================================================

def parse_json_robust(raw: str) -> dict:
    """
    Robustly extract a JSON object from a possibly noisy string.
    """
    raw = raw.strip()
    # Remove typical markdown fences if present
    if raw.startswith("```"):
        # Strip leading/trailing fences
        raw = raw.strip("`")
        # Sometimes still starts with json\n
        if raw.lower().startswith("json"):
            raw = raw[4:].lstrip()

    # Direct attempt
    try:
        return json.loads(raw)
    except Exception:
        pass

    # Fallback: take substring between first '{' and last '}'
    first = raw.find("{")
    last = raw.rfind("}")
    if first != -1 and last != -1 and last > first:
        snippet = raw[first:last + 1]
        try:
            return json.loads(snippet)
        except Exception:
            raise HTTPException(500, f"Model response is not valid JSON: {raw[:200]}")
    raise HTTPException(500, f"Model response is not valid JSON: {raw[:200]}")


def groq_call(model: str, system: str, user_blocks: List[dict]) -> Tuple[dict, TokenUsage]:
    try:
        resp = client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system}],
                },
                {
                    "role": "user",
                    "content": user_blocks,
                },
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
    parsed = parse_json_robust(raw)
    return parsed, tu


def normalize_pages_from_fast(parsed: Any, num_pages: int) -> Dict[int, Dict[str, Any]]:
    """
    Normalize the FAST (Scout) model output into a dict:
        index -> {page_no, page_type, bill_items}
    """

    if isinstance(parsed, dict) and "pagewise_line_items" in parsed:
        raw_pages = parsed.get("pagewise_line_items") or []
    elif isinstance(parsed, list):
        raw_pages = parsed
    else:
        raw_pages = []

    pages_dict: Dict[int, Dict[str, Any]] = {
        i: {"page_no": str(i + 1), "page_type": "Bill Detail", "bill_items": []}
        for i in range(num_pages)
    }

    for p in raw_pages:
        if not isinstance(p, dict):
            continue

        page_no_raw = p.get("page_no", None)
        idx = None
        try:
            idx = int(str(page_no_raw).strip()) - 1
        except Exception:
            # ignore invalid page_no
            continue

        if idx is None or idx < 0 or idx >= num_pages:
            continue

        pages_dict[idx] = p

    return pages_dict


def normalize_single_page_output(parsed: Any, page_index: int) -> Dict[str, Any]:
    """
    Normalize ACCURATE (Maverick) output for a single page into a page dict.
    If it returns a multi-page object, pick appropriate page.
    """
    default_page_no = str(page_index + 1)

    if isinstance(parsed, dict):
        if "pagewise_line_items" in parsed:
            candidates = parsed.get("pagewise_line_items") or []
            chosen = None
            # Prefer page with matching page_no
            for p in candidates:
                if isinstance(p, dict) and str(p.get("page_no")) == default_page_no:
                    chosen = p
                    break
            if chosen is None and candidates:
                chosen = candidates[0]
            page = chosen or {}
        else:
            page = parsed
    elif isinstance(parsed, list):
        page = parsed[0] if parsed else {}
    else:
        page = {}

    # Enforce required shape
    if not isinstance(page, dict):
        page = {}
    page.setdefault("page_no", default_page_no)
    page.setdefault("page_type", "Final Bill")
    page.setdefault("bill_items", [])

    return page


# ============================================================
#  Cleaning
# ============================================================

def norm_num(x: Any) -> float:
    if x is None:
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.replace(",", "").strip()
        if s in ["", "-", "—", "NA", "N/A"]:
            return 0.0
        try:
            return float(s)
        except Exception:
            return 0.0
    return 0.0


def clean_page_dict(p: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pre-clean and dedupe within a single page dict.
    """
    items: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, float, float, float]] = set()

    for it in (p.get("bill_items") or []):
        if not isinstance(it, dict):
            continue

        name = str(it.get("item_name", "")).strip()
        amt = norm_num(it.get("item_amount"))
        rate = norm_num(it.get("item_rate"))
        qty = norm_num(it.get("item_quantity"))

        key = (name.lower(), round(amt, 2), round(rate, 4), round(qty, 4))
        if key in seen:
            continue
        seen.add(key)

        items.append(
            {
                "item_name": name,
                "item_amount": amt,
                "item_rate": rate,
                "item_quantity": qty,
            }
        )

    return {
        "page_no": str(p.get("page_no", "")),
        "page_type": str(p.get("page_type", "Bill Detail")),
        "bill_items": items,
    }


def fix_math(page: PageItems) -> PageItems:
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


def dedupe_across_pages(pages: List[PageItems]) -> List[PageItems]:
    """
    De-duplicate clearly repeated items across pages.
    """
    seen: Set[Tuple[str, str, float, float, float]] = set()
    out: List[PageItems] = []

    for p in pages:
        new_items: List[BillItem] = []
        for it in p.bill_items:
            key = (
                p.page_type.strip().lower(),
                it.item_name.strip().lower(),
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
    return {
        "status": "OK",
        "message": "Use POST /extract-bill-data with JSON body "
                   '{"document": "<public image/PDF URL>"}',
    }


# ============================================================
#  POST Endpoint
# ============================================================

@app.post("/extract-bill-data", response_model=ExtractBillDataResponse)
def extract_api(req: ExtractBillDataRequest):
    t0 = time.time()
    doc_url = str(req.document)

    # 1) Download
    content = download_document(doc_url)

    # 2) Convert to images (one per page)
    images = document_to_images(doc_url, content)
    num_pages = len(images)

    if num_pages == 0:
        return ExtractBillDataResponse(
            is_success=False,
            message="No pages/images could be extracted from document.",
        )

    # --------------------------------------------------------
    # 1) FAST BATCH PASS (Scout) — <= 5 images per call
    # --------------------------------------------------------
    batches: List[List[dict]] = []
    for i in range(0, num_pages, 5):
        block: List[dict] = [
            {
                "type": "input_text",
                "text": f"Pages {i + 1}-{min(i + 5, num_pages)}:",
            }
        ]
        for j in range(i, min(i + 5, num_pages)):
            block.append(
                {
                    "type": "input_text",
                    "text": f"PAGE {j + 1}:",
                }
            )
            block.append(
                {
                    "type": "input_image",
                    "image_url": images[j],
                    "detail": "auto",
                }
            )
        batches.append(block)

    total_usage = TokenUsage(0, 0, 0)
    pages_dict: Dict[int, Dict[str, Any]] = {
        i: {"page_no": str(i + 1), "page_type": "Bill Detail", "bill_items": []}
        for i in range(num_pages)
    }

    for b in batches:
        user_blocks = [{"type": "input_text", "text": USER_PROMPT_GLOBAL}] + b
        parsed, u = groq_call(MODEL_FAST, SYSTEM_PROMPT, user_blocks)

        total_usage.total_tokens += u.total_tokens
        total_usage.input_tokens += u.input_tokens
        total_usage.output_tokens += u.output_tokens

        part_pages = normalize_pages_from_fast(parsed, num_pages)
        # Merge into global dict (later batches can overwrite same pages)
        for idx, page_dict in part_pages.items():
            pages_dict[idx] = page_dict

    # --------------------------------------------------------
    # 2) ACCURACY PASS (Maverick) on Final Bill pages only
    # --------------------------------------------------------
    for idx in range(num_pages):
        p = pages_dict[idx]
        ptype = str(p.get("page_type", "")).strip().lower()
        if ptype == "final bill":
            # Re-run just this page with Maverick, more accurate read
            parsed, u = groq_call(
                MODEL_ACCURATE,
                SYSTEM_PROMPT,
                [
                    {"type": "input_text", "text": USER_PROMPT_SINGLE},
                    {
                        "type": "input_image",
                        "image_url": images[idx],
                        "detail": "high",
                    },
                ],
            )
            total_usage.total_tokens += u.total_tokens
            total_usage.input_tokens += u.input_tokens
            total_usage.output_tokens += u.output_tokens

            page_norm = normalize_single_page_output(parsed, idx)
            pages_dict[idx] = page_norm

    # --------------------------------------------------------
    # 3) CLEANING + VALIDATION
    # --------------------------------------------------------
    pages: List[PageItems] = []
    for idx in range(num_pages):
        raw_page = pages_dict[idx]
        # ensure correct page_no
        raw_page["page_no"] = str(idx + 1)
        cleaned_dict = clean_page_dict(raw_page)
        try:
            page = PageItems(**cleaned_dict)
        except Exception as e:
            raise HTTPException(500, f"Model JSON does not match schema: {e}")
        page = fix_math(page)
        pages.append(page)

    # 4) De-duplicate across pages
    pages = dedupe_across_pages(pages)

    # 5) Aggregate
    total_items = sum(len(p.bill_items) for p in pages)
    data = ExtractBillDataResponseData(
        pagewise_line_items=pages,
        total_item_count=total_items,
    )

    elapsed = time.time() - t0
    print(
        f"[BILL] pages={num_pages} items={total_items} "
        f"tokens={total_usage.total_tokens} time={elapsed:.2f}s"
    )

    return ExtractBillDataResponse(
        is_success=True,
        token_usage=total_usage,
        data=data,
    )
