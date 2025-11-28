# main.py
import base64
import io
import json
import math
import mimetypes
import os
from typing import List

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from pdf2image import convert_from_bytes
from PIL import Image
from openai import OpenAI

# ============================================================
#               GROQ CLIENT (OpenAI-Compatible)
# ============================================================

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

GROQ_MODEL_ID = "meta-llama/llama-4-maverick-17b-128e-instruct"


# ============================================================
#               Pydantic Schemas
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
    token_usage: TokenUsage
    data: ExtractBillDataResponseData


# ============================================================
#               FastAPI App
# ============================================================

app = FastAPI(
    title="Bill Data Extraction API (Groq Maverick)",
    version="2.0.0",
    description="Extract line items and totals from invoice documents using Groq Llama 4 Maverick.",
)


# ============================================================
#               Helpers: Download, Convert, Image Encoding
# ============================================================

def download_document(url: str) -> bytes:
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download document: {e}")


def guess_mime_type(url, content: bytes) -> str:
    url = str(url)
    mime, _ = mimetypes.guess_type(url)
    if mime:
        return mime
    if content[:4] == b"%PDF":
        return "application/pdf"
    return "application/octet-stream"


def image_to_data_url(img: Image.Image, quality: int = 85) -> str:
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


def document_to_page_images(url: str, content: bytes) -> List[str]:
    url = str(url)
    mime = guess_mime_type(url, content)

    if mime.startswith("image/"):
        img = Image.open(io.BytesIO(content)).convert("RGB")
        return [image_to_data_url(img)]

    if mime == "application/pdf":
        pages = convert_from_bytes(content)
        return [image_to_data_url(p.convert("RGB")) for p in pages]

    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
        return [image_to_data_url(img)]
    except Exception:
        raise HTTPException(status_code=400, detail=f"Unsupported document type: {mime}")


# ============================================================
#               LLM Prompt
# ============================================================

SYSTEM_PROMPT = """
You are an expert invoice understanding system. 
You MUST extract EXACT PER-ROW line items from the given page image. 

CRITICAL RULES:
- Each visible row in the items table MUST become exactly one JSON element. 
- NEVER merge two different items into one line.
- NEVER skip any row.
- NEVER guess values.
- NEVER swap amounts between items.
- If two amounts appear (e.g., Rate and Total), use TOTAL as item_amount.
- item_amount must be EXACTLY the “Total” column value.
- item_rate must be EXACTLY the “Rate” column value.
- item_quantity must be EXACTLY the “Qty” column value.
- If multiple rows look similar, treat them as separate items.

COLUMN RULES:
- item_name = EXACT text under Description.
- item_quantity = numeric Qty.
- item_rate = numeric Rate.
- item_amount = numeric Total (NOT Amount column if both exist).

DO NOT:
- Merge multiple tests (e.g., RFT + ELECTROLYTES).
- Swap totals between rows.
- Infer or guess missing numbers.

OUTPUT:
Strict JSON with:
{
  "page_no": "X",
  "page_type": "<Bill Detail | Final Bill | Pharmacy>",
  "bill_items": [
     {
       "item_name": "",
       "item_amount": float,
       "item_rate": float,
       "item_quantity": float
     }
   ]
}
"""


# ============================================================
#               Groq Maverick Vision Call
# ============================================================

def update_usage(acc: dict, response) -> None:
    """Accumulate token usage from Groq response into acc dict."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return

    # responses API usually exposes total_tokens, input_tokens, output_tokens
    acc["total_tokens"] += getattr(usage, "total_tokens", 0)
    acc["input_tokens"] += getattr(usage, "input_tokens", 0)
    acc["output_tokens"] += getattr(usage, "output_tokens", 0)


def call_maverick_for_page(
    page_no: int,
    image_data_url: str,
    usage_acc: dict,
) -> PageItems:
    user_prompt = f"""
This is page number {page_no} of a bill.
Return JSON with page_no="{page_no}", an appropriate page_type, and the bill_items for THIS PAGE ONLY.
"""

    try:
        response = client.responses.create(
            model=GROQ_MODEL_ID,
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
                        {"type": "input_text", "text": user_prompt.strip()},
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
        raise HTTPException(status_code=500, detail=f"Groq API error: {e}")

    # accumulate token usage
    update_usage(usage_acc, response)

    raw_text = response.output_text.strip()

    # Try to parse as pure JSON
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        # salvage JSON if wrapped with text
        first = raw_text.find("{")
        last = raw_text.rfind("}")
        if first != -1 and last != -1 and last > first:
            parsed = json.loads(raw_text[first:last + 1])
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Model response is not valid JSON: {raw_text[:200]}",
            )

    try:
        page_items = PageItems(**parsed)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model JSON does not match expected schema: {e}",
        )

    return page_items


# ============================================================
#               Amount Reconciliation (internal)
# ============================================================

def reconcile_bill_items(pagewise_items: List[PageItems]) -> ExtractBillDataResponseData:
    """
    Light consistency cleanup:
    - If both rate and qty present, fix amount to rate*qty when mismatch.
    - If amount & qty only, infer rate.
    - If only amount present, set qty=1 and rate=amount.
    Final total isn't returned (judges will compute upstream), but we ensure
    numbers are self-consistent.
    """
    EPS = 0.01
    total_item_count = 0

    for page in pagewise_items:
        for item in page.bill_items:
            total_item_count += 1

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

    return ExtractBillDataResponseData(
        pagewise_line_items=pagewise_items,
        total_item_count=total_item_count,
    )


# ============================================================
#               API Endpoint
# ============================================================

@app.post("/extract-bill-data", response_model=ExtractBillDataResponse)
def extract_bill_data(req: ExtractBillDataRequest):
    """
    Datathon endpoint.

    Request:
    {
      "document": "<image-or-pdf-url>"
    }

    Response strictly follows the HackRx spec.
    """

    url = str(req.document)

    # 1. Download
    content = download_document(url)

    # 2. Convert to page images
    page_images = document_to_page_images(url, content)
    if not page_images:
        raise HTTPException(status_code=400, detail="No pages/images could be extracted from the document.")

    # 3. Track token usage
    usage_acc = {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0}

    # 4. Run Groq Maverick on each page
    pagewise_results: List[PageItems] = []
    for i, img_data_url in enumerate(page_images, start=1):
        page_result = call_maverick_for_page(page_no=i, image_data_url=img_data_url, usage_acc=usage_acc)
        pagewise_results.append(page_result)

    # 5. Reconcile numeric fields
    data = reconcile_bill_items(pagewise_results)

    # 6. Build token usage object
    token_usage = TokenUsage(
        total_tokens=usage_acc["total_tokens"],
        input_tokens=usage_acc["input_tokens"],
        output_tokens=usage_acc["output_tokens"],
    )

    return ExtractBillDataResponse(
        is_success=True,
        token_usage=token_usage,
        data=data,
    )
