# ============================================================
# Bajaj Datathon – Hybrid Vision + OCR Bill Extraction API
# Final main.py – Adaptive Hybrid Prompt, CPU OCR, Multipage-safe
#
# Requirements (add to requirements.txt):
# fastapi
# uvicorn
# httpx
# pydantic
# pillow
# pdf2image
# opencv-python
# numpy
# pytesseract
# paddlepaddle
# paddleocr
# openai
#
# Run locally:
#   export GROQ_API_KEY="your_key_here"
#   uvicorn main:app --reload
# ============================================================

import os
import io
import re
import cv2
import json
import math
import base64
import mimetypes
import asyncio
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from PIL import Image
from pdf2image import convert_from_bytes

import pytesseract
from paddleocr import PaddleOCR
from openai import AsyncOpenAI

# ============================================================
# CONFIG & CLIENTS
# ============================================================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable is not set.")

# Groq async client
groq = AsyncOpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

# Vision models
MODEL_STRONG = "meta-llama/llama-4-maverick-17b-128e-instruct"
MODEL_FALLBACK = "meta-llama/llama-4-scout-17b-16e-instruct"

# CPU PaddleOCR (no GPU)
paddle_ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en",
    show_log=False
)

# ============================================================
# Pydantic Schemas – EXACT Datathon Spec
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
# FastAPI App
# ============================================================

app = FastAPI(
    title="Bajaj Datathon – Hybrid Bill Extraction API",
    version="4.1.0",
    description=(
        "Vision + PaddleOCR + Tesseract + Adaptive Prompting + Async Groq\n"
        "Implements the exact HackRx Datathon schema for bill line-item extraction."
    ),
)

# ============================================================
# Utility Helpers – Download & Image Handling
# ============================================================

async def download_document(url: str) -> bytes:
    """Download document (PDF/image) via HTTP."""
    try:
        async with httpx.AsyncClient(timeout=40) as client:
            resp = await client.get(url)
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


def preprocess_image(img: Image.Image) -> Image.Image:
    """
    Light preprocessing: deskew + slight sharpen.
    Keeps it robust for both clean and noisy bills.
    """
    cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)

    # Deskew based on darkest pixels (text/lines)
    coords = np.column_stack(np.where(gray < 200))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle += 90
        M = cv2.getRotationMatrix2D((cv.shape[1] // 2, cv.shape[0] // 2), angle, 1.0)
        cv = cv2.warpAffine(cv, M, (cv.shape[1], cv.shape[0]), flags=cv2.INTER_CUBIC)

    # Slight sharpen
    blur = cv2.GaussianBlur(cv, (0, 0), 3)
    cv = cv2.addWeighted(cv, 1.5, blur, -0.5, 0)

    return Image.fromarray(cv2.cvtColor(cv, cv2.COLOR_BGR2RGB))


def image_to_b64(img: Image.Image) -> str:
    """Encode PIL image as base64 JPEG data URL (<=4MB)."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    data = buf.getvalue()
    quality = 85
    while len(data) > 4 * 1024 * 1024 and quality > 30:
        quality -= 10
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        data = buf.getvalue()
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("utf-8")


def hybrid_ocr(img: Image.Image) -> str:
    """
    Combine Tesseract + PaddleOCR text for stronger signals.
    We use text only (no boxes) to keep pipeline simple & fast.
    """
    try:
        tess_text = pytesseract.image_to_string(img) or ""
    except Exception:
        tess_text = ""

    paddle_texts: List[str] = []
    try:
        res = paddle_ocr.ocr(np.array(img), cls=True)
        if res:
            for block in res:
                for line in block:
                    paddle_texts.append(line[1][0])
    except Exception:
        pass

    full_text = (tess_text + "\n" + "\n".join(paddle_texts)).strip()
    return full_text


def classify_page_type(ocr_text: str) -> str:
    """
    Simple heuristic classifier using OCR text.
    """
    t = (ocr_text or "").lower()
    if any(x in t for x in ["pharmacy", "pharma", "tablet", "capsule", "inj", "syrup", "tab "]):
        return "Pharmacy"
    if "final bill" in t or "net payable" in t or "summary" in t:
        return "Final Bill"
    return "Bill Detail"


def generate_crops(img: Image.Image, n: int = 3) -> List[Image.Image]:
    """
    Vertical crops to improve recall on long tables.
    """
    w, h = img.size
    if n <= 1:
        return [img]

    crops: List[Image.Image] = []
    step = h // n
    for i in range(n):
        top = i * step
        bottom = h if i == n - 1 else (i + 1) * step
        crops.append(img.crop((0, top, w, bottom)))
    return crops


# ============================================================
# Adaptive Hybrid Prompt (Few-shot per Page Type)
# ============================================================

BASE_SYSTEM_PROMPT = """
You are a highly accurate medical bill line-item extraction engine.

Your ONLY goal:
From ONE page image, extract EVERY visible LINE-ITEM row and RETURN STRICT JSON.

A "line item" is a row in a table with:
- item_name  (description of test/procedure/medicine/room/charge)
- item_quantity
- item_rate
- item_amount

DO NOT extract:
- totals (SUB TOTAL, GRAND TOTAL, NET AMOUNT, AMOUNT PAYABLE, ROUND OFF, TAX, CGST, SGST, IGST)
- discounts (DISC, DISCOUNT)
- section headers (PHARMACY, RADIOLOGY, INVESTIGATION, CONSULTATION, OTHERS, PACKAGE, CHARGES)
- patient details (name, age, sex, IP number, doctor name)
- address, phone numbers, disclaimers, notes

DO NOT hallucinate any item.
Every output row MUST correspond to a real row visible in THIS PAGE IMAGE.
Do NOT pull items from previous or next pages.

If some numeric fields are missing:
- If only a single amount is visible for that row:
    → item_quantity = 1.0
    → item_rate = item_amount
- If quantity and amount visible, but rate missing:
    → item_rate = item_amount / item_quantity
- If rate and amount visible, but quantity missing:
    → item_quantity = item_amount / item_rate
Use 0.0 ONLY when the numeric field is truly not present in the row.

JSON Schema (exact):

{
  "page_no": "<string>",
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

Rules:
- Copy item_name exactly from the row (ignore batch no/expiry if they clutter).
- Use dot (.) as decimal separator.
- Do NOT wrap JSON in ``` fences.
- Do NOT add any extra top-level keys.
"""


PHARMACY_FEWSHOT = """
PAGE TYPE: Pharmacy

EXAMPLE 1 (Pharmacy)
IMAGE (conceptual):
Qty | Name              | Rate  | Amount
1   | Paracetamol 650   | 25.00 | 25.00
2   | ORS 500ML         | 40.00 | 80.00

CORRECT JSON:
{
  "page_no": "1",
  "page_type": "Pharmacy",
  "bill_items": [
    {"item_name": "Paracetamol 650", "item_amount": 25.0, "item_rate": 25.0, "item_quantity": 1.0},
    {"item_name": "ORS 500ML", "item_amount": 80.0, "item_rate": 40.0, "item_quantity": 2.0}
  ]
}

EXAMPLE 2 (Missing Qty, only amount)
IMAGE:
Name                 | Amount
DOLO 650 TAB (strip) | 45.00

CORRECT JSON:
{
  "page_no": "1",
  "page_type": "Pharmacy",
  "bill_items": [
    {"item_name": "DOLO 650 TAB (strip)", "item_amount": 45.0, "item_rate": 45.0, "item_quantity": 1.0}
  ]
}
"""


BILL_DETAIL_FEWSHOT = """
PAGE TYPE: Bill Detail (investigations / services)

EXAMPLE 1
IMAGE:
SERVICE                  | Qty | Rate    | Total
X-Ray                    | 1   | 500.00  | 500.00
Registration             | 1   | 300.00  | 300.00
Room Charges             | 3   | 2000.00 | 6000.00

CORRECT JSON:
{
  "page_no": "1",
  "page_type": "Bill Detail",
  "bill_items": [
    {"item_name": "X-Ray", "item_amount": 500.0, "item_rate": 500.0, "item_quantity": 1.0},
    {"item_name": "Registration", "item_amount": 300.0, "item_rate": 300.0, "item_quantity": 1.0},
    {"item_name": "Room Charges", "item_amount": 6000.0, "item_rate": 2000.0, "item_quantity": 3.0}
  ]
}

EXAMPLE 2 (Missing Qty)
IMAGE:
Nebulization Charge | Rate 100.00 | Amount 1000.00

CORRECT JSON:
{
  "page_no": "1",
  "page_type": "Bill Detail",
  "bill_items": [
    {"item_name": "Nebulization Charge", "item_amount": 1000.0, "item_rate": 100.0, "item_quantity": 10.0}
  ]
}
"""


FINAL_BILL_FEWSHOT = """
PAGE TYPE: Final Bill (summary-style)

EXAMPLE (Final summary)
IMAGE:
DETAILS                   | Qty | Rate  | Amount
Package Charge            | 1   | 50000 | 50000
Nursing Charges           | 3   | 500   | 1500

SUB TOTAL                 | 51500
CGST 9%                   | 4635
SGST 9%                   | 4635
NET AMOUNT PAYABLE        | 60770

CORRECT JSON (notice total rows excluded):
{
  "page_no": "1",
  "page_type": "Final Bill",
  "bill_items": [
    {"item_name": "Package Charge", "item_amount": 50000.0, "item_rate": 50000.0, "item_quantity": 1.0},
    {"item_name": "Nursing Charges", "item_amount": 1500.0, "item_rate": 500.0, "item_quantity": 3.0}
  ]
}
"""


def build_system_prompt(page_type: str) -> str:
    """
    Return adaptive system prompt with appropriate few-shots.
    """
    base = BASE_SYSTEM_PROMPT.strip()
    pt = (page_type or "Bill Detail").lower()

    if "pharmacy" in pt:
        return base + "\n\n" + PHARMACY_FEWSHOT.strip()
    if "final" in pt:
        return base + "\n\n" + FINAL_BILL_FEWSHOT.strip()
    return base + "\n\n" + BILL_DETAIL_FEWSHOT.strip()


# ============================================================
# LLM Call – with adaptive prompt + fallback model
# ============================================================

async def call_llm_for_crop(
    img_b64: str,
    page_no: int,
    page_type: str,
    ocr_text: str,
    model_name: str,
) -> Tuple[Dict[str, Any], TokenUsage]:
    """
    Call Groq vision model for one crop with adaptive system prompt.
    """

    system_prompt = build_system_prompt(page_type)
    ocr_snippet = (ocr_text or "").strip()
    if len(ocr_snippet) > 1600:
        ocr_snippet = ocr_snippet[:1600]

    user_text = f"""
You are extracting line items from ONE crop of page {page_no} of a medical bill.

This page has type: {page_type}.

Below is OCR text from THIS PAGE. Use it as a hint for small numbers and text,
but always align with what you see in the image crop.

OCR_TEXT_START
{ocr_snippet}
OCR_TEXT_END

Now return ONLY the JSON for this crop, following the schema.
Do not include subtotals or taxes as items.
""".strip()

    try:
        resp = await groq.responses.create(
            model=model_name,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_text},
                        {
                            "type": "input_image",
                            "image_url": img_b64,
                            "detail": "auto",
                        },
                    ],
                },
            ],
        )
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Groq API error ({model_name}): {e}",
        )

    usage_obj = resp.usage
    token_usage = TokenUsage(
        total_tokens=int(getattr(usage_obj, "total_tokens", 0) or 0),
        input_tokens=int(getattr(usage_obj, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage_obj, "output_tokens", 0) or 0),
    )

    raw_text = resp.output_text.strip()

    # Robust JSON parsing
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        first = raw_text.find("{")
        last = raw_text.rfind("}")
        if first != -1 and last != -1 and last > first:
            json_str = raw_text[first:last + 1]
            parsed = json.loads(json_str)
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Model response not valid JSON: {raw_text[:200]}",
            )

    return parsed, token_usage


async def extract_items_from_crop(
    crop: Image.Image,
    page_no: int,
    page_type: str,
    ocr_text: str,
) -> Tuple[List[Dict[str, Any]], TokenUsage]:
    """
    One crop → try Maverick, fallback Scout.
    Returns bill_items list + token usage.
    """
    img_b64 = image_to_b64(crop)
    # Heuristic: Pharmacy or very noisy text → Scout; else Maverick.
    t = (page_type or "").lower()
    if "pharmacy" in t:
        primary, backup = MODEL_FALLBACK, MODEL_STRONG
    else:
        primary, backup = MODEL_STRONG, MODEL_FALLBACK

    # Try primary
    try:
        result, usage = await call_llm_for_crop(
            img_b64=img_b64,
            page_no=page_no,
            page_type=page_type,
            ocr_text=ocr_text,
            model_name=primary,
        )
        items = result.get("bill_items", []) or []
        return items, usage
    except Exception:
        # Fallback
        result, usage = await call_llm_for_crop(
            img_b64=img_b64,
            page_no=page_no,
            page_type=page_type,
            ocr_text=ocr_text,
            model_name=backup,
        )
        items = result.get("bill_items", []) or []
        return items, usage


# ============================================================
# Numeric & Dedupe Helpers
# ============================================================

def _coerce_number(x: Any) -> float:
    if x is None:
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == "" or s in {"-", "—", "NA", "N/A"}:
        return 0.0
    s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return 0.0


def reconcile_item_numeric(it: Dict[str, Any]) -> Dict[str, Any]:
    name = str(it.get("item_name", "")).strip()
    amt = _coerce_number(it.get("item_amount"))
    rate = _coerce_number(it.get("item_rate"))
    qty = _coerce_number(it.get("item_quantity"))

    # Basic inference rules
    if rate and qty and not amt:
        amt = round(rate * qty, 2)
    elif amt and qty and not rate:
        rate = round(amt / qty, 4)
    elif amt and rate and not qty:
        qty = round(amt / rate, 4)
    elif amt and (not rate and not qty):
        qty = 1.0
        rate = amt

    # If still all zero, keep them zero
    return {
        "item_name": name,
        "item_amount": float(amt),
        "item_rate": float(rate),
        "item_quantity": float(qty),
    }


def looks_like_total_row(name: str) -> bool:
    low = name.lower()
    keywords = [
        "total",
        "sub total",
        "subtotal",
        "net amount",
        "amount payable",
        "round off",
        "roundoff",
        "cgst",
        "sgst",
        "igst",
        "gst",
        "tax",
        "discount",
        "disc",
    ]
    return any(kw in low for kw in keywords)


def semantic_dedupe_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Simple semantic dedupe: same normalized name and near-equal numbers.
    Good enough for multi-crop & multi-page duplicates.
    """
    seen = set()
    out: List[Dict[str, Any]] = []
    for it in items:
        key = (
            it["item_name"].strip().lower(),
            round(it["item_amount"], 2),
            round(it["item_rate"], 4),
            round(it["item_quantity"], 4),
        )
        if key not in seen:
            seen.add(key)
            out.append(it)
    return out


# ============================================================
# Per-page Processing
# ============================================================

async def process_page(img: Image.Image, page_no: int) -> Tuple[PageItems, TokenUsage]:
    """
    Process a SINGLE page:
    - Preprocess image
    - OCR
    - Page-type classification
    - Multi-crop + async LLM calls
    - Merge, reconcile, dedupe
    """
    img = preprocess_image(img)
    ocr_text = hybrid_ocr(img)
    page_type = classify_page_type(ocr_text)

    crops = generate_crops(img, n=3 if page_type != "Pharmacy" else 4)

    tasks = [
        extract_items_from_crop(crop, page_no=page_no, page_type=page_type, ocr_text=ocr_text)
        for crop in crops
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_items_raw: List[Dict[str, Any]] = []
    total_tokens = input_tokens = output_tokens = 0

    for res in results:
        if isinstance(res, Exception):
            continue
        items, usage = res
        total_tokens += usage.total_tokens
        input_tokens += usage.input_tokens
        output_tokens += usage.output_tokens
        for it in items:
            # Reconcile numeric + filter totals
            rec = reconcile_item_numeric(it)
            if not rec["item_name"]:
                continue
            if looks_like_total_row(rec["item_name"]):
                continue
            all_items_raw.append(rec)

    # Dedupe within page
    all_items = semantic_dedupe_items(all_items_raw)

    page_obj = PageItems(
        page_no=str(page_no),
        page_type=page_type,
        bill_items=[
            BillItem(
                item_name=it["item_name"],
                item_amount=it["item_amount"],
                item_rate=it["item_rate"],
                item_quantity=it["item_quantity"],
            )
            for it in all_items
        ],
    )

    usage_page = TokenUsage(
        total_tokens=total_tokens,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    return page_obj, usage_page


# ============================================================
# API Endpoint
# ============================================================

@app.post("/extract-bill-data", response_model=ExtractBillDataResponse)
async def extract_bill_data(req: ExtractBillDataRequest):
    """
    Datathon Endpoint:
      POST /extract-bill-data
      Body: { "document": "<public PDF/image URL>" }
    """
    url_str = str(req.document)

    # 1. Download document
    content = await download_document(url_str)
    mime = guess_mime_type(url_str, content)

    # 2. Convert to pages
    if mime == "application/pdf":
        try:
            pil_pages = convert_from_bytes(content)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Unable to convert PDF to images. Is poppler installed? Error: {e}",
            )
    else:
        try:
            img = Image.open(io.BytesIO(content)).convert("RGB")
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Unable to open image document: {e}",
            )
        pil_pages = [img]

    if not pil_pages:
        return ExtractBillDataResponse(
            is_success=False,
            message="No pages could be extracted from the document.",
        )

    # 3. Process all pages sequentially (can be parallelized if needed)
    all_pages: List[PageItems] = []
    total_tokens = input_tokens = output_tokens = 0

    for page_idx, page_img in enumerate(pil_pages, start=1):
        page_items, usage = await process_page(page_img, page_no=page_idx)
        all_pages.append(page_items)
        total_tokens += usage.total_tokens
        input_tokens += usage.input_tokens
        output_tokens += usage.output_tokens

    # 4. Global semantic dedupe across pages (prevent double counting)
    #    This is conservative: we only dedupe exact clones.
    global_seen = set()
    for p in all_pages:
        new_items = []
        for it in p.bill_items:
            key = (
                it.item_name.strip().lower(),
                round(it.item_amount, 2),
                round(it.item_rate, 4),
                round(it.item_quantity, 4),
            )
            if key not in global_seen:
                global_seen.add(key)
                new_items.append(it)
        p.bill_items = new_items

    total_item_count = sum(len(p.bill_items) for p in all_pages)

    token_usage = TokenUsage(
        total_tokens=total_tokens,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    data = ExtractBillDataResponseData(
        pagewise_line_items=all_pages,
        total_item_count=total_item_count,
    )

    return ExtractBillDataResponse(
        is_success=True,
        token_usage=token_usage,
        data=data,
    )
