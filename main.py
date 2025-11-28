# main.py
import base64
import io
import json
import math
import mimetypes
import os
from typing import List, Optional

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
    base_url="https://api.groq.com/openai/v1",   # Groq endpoint
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
    bill_items: List[BillItem]


class ExtractBillDataRequest(BaseModel):
    document: HttpUrl   # comes as pydantic HttpUrl


class ExtractBillDataResponseData(BaseModel):
    pagewise_line_items: List[PageItems]
    total_item_count: int
    reconciled_amount: float


class ExtractBillDataResponse(BaseModel):
    is_success: bool
    data: Optional[ExtractBillDataResponseData] = None
    error_message: Optional[str] = None


# ============================================================
#               FastAPI App
# ============================================================

app = FastAPI(
    title="Bill Data Extraction API (Groq Maverick)",
    version="1.0.1",
    description="Extract line items and totals from invoice documents using Groq Llama 4 Maverick.",
)


# ============================================================
#               Helpers: Download, Convert, Image Encoding
# ============================================================

def download_document(url: str) -> bytes:
    """Download the document from the given URL."""
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download document: {e}")


def guess_mime_type(url, content: bytes) -> str:
    """Accepts HttpUrl or str, returns MIME."""
    url = str(url)  # FIX
    mime, _ = mimetypes.guess_type(url)
    if mime:
        return mime
    if content[:4] == b"%PDF":
        return "application/pdf"
    return "application/octet-stream"


def image_to_data_url(img: Image.Image, quality: int = 85) -> str:
    """Convert PIL image to Base64 < 4MB for Groq."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b = buf.getvalue()

    # Reduce quality until under 4MB
    while len(b) > 4 * 1024 * 1024 and quality > 30:
        quality -= 10
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        b = buf.getvalue()

    b64 = base64.b64encode(b).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def document_to_page_images(url: str, content: bytes) -> List[str]:
    """Convert PDF or image to a list of image data URLs."""
    url = str(url)  # FIX
    mime = guess_mime_type(url, content)

    # Single image file
    if mime.startswith("image/"):
        img = Image.open(io.BytesIO(content)).convert("RGB")
        return [image_to_data_url(img)]

    # Multi-page PDF
    if mime == "application/pdf":
        pages = convert_from_bytes(content)
        return [image_to_data_url(p.convert("RGB")) for p in pages]

    # Try image anyway
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
        return [image_to_data_url(img)]
    except Exception:
        raise HTTPException(status_code=400, detail=f"Unsupported document type: {mime}")


# ============================================================
#               LLM Prompt
# ============================================================

SYSTEM_PROMPT = """
You are an expert system for extracting structured data from invoices and bills.

Given a SINGLE PAGE image of an invoice, extract ONLY the LINE ITEMS as strict JSON.

Output exactly:
{
  "page_no": "<string>",
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
- Only line items (product rows).
- Ignore totals, subtotals, GST, discounts, doctor names, etc.
- item_amount = item_rate * item_quantity if possible.
- If quantity missing → set quantity to 1.
- DO NOT add commentary.
- DO NOT add extra keys.
"""


# ============================================================
#               Groq Maverick Vision Call
# ============================================================

def call_maverick_for_page(page_no: int, image_data_url: str) -> PageItems:
    """Send a single invoice page image to Groq Maverick Vision."""

    user_prompt = f"""
Extract line items for page number {page_no}.
Use page_no="{page_no}".
"""

    try:
        response = client.responses.create(
            model=GROQ_MODEL_ID,
            input=[
                {
                    "role": "system",
                    "content": [
                        {"type": "input_text", "text": SYSTEM_PROMPT},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_prompt},
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

    raw_text = response.output_text.strip()

    # Try strict JSON first
    try:
        parsed = json.loads(raw_text)
    except:
        # Try to salvage JSON inside text
        first = raw_text.find("{")
        last = raw_text.rfind("}")
        if first != -1 and last != -1:
            parsed = json.loads(raw_text[first:last+1])
        else:
            raise HTTPException(status_code=500, detail=f"Invalid JSON from model: {raw_text[:200]}")

    try:
        return PageItems(**parsed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"JSON schema mismatch: {e}")


# ============================================================
#               Amount Reconciliation
# ============================================================

def reconcile_bill_items(pagewise_items: List[PageItems]) -> ExtractBillDataResponseData:
    total_item_count = 0
    total_amount = 0.0
    EPS = 0.01

    for page in pagewise_items:
        for item in page.bill_items:
            total_item_count += 1

            amount = float(item.item_amount)
            rate = float(item.item_rate)
            qty = float(item.item_quantity)

            # Fix inconsistencies
            if rate and qty:
                computed = rate * qty
                if abs(computed - amount) > EPS:
                    item.item_amount = round(computed, 2)
            elif amount and qty:
                item.item_rate = round(amount / qty, 4)
            elif amount:
                item.item_quantity = 1.0
                item.item_rate = round(amount, 2)

            total_amount += item.item_amount

    return ExtractBillDataResponseData(
        pagewise_line_items=pagewise_items,
        total_item_count=total_item_count,
        reconciled_amount=round(total_amount, 2),
    )


# ============================================================
#               API Endpoint
# ============================================================

@app.post("/extract-bill-data", response_model=ExtractBillDataResponse)
def extract_bill_data(req: ExtractBillDataRequest):
    """Main Datathon API."""

    url = str(req.document)  # FIX

    # Download invoice
    content = download_document(url)

    # Convert invoice to images
    page_images = document_to_page_images(url, content)

    pagewise_results = []
    for i, img in enumerate(page_images, start=1):
        pagewise_results.append(call_maverick_for_page(i, img))

    # Reconcile totals
    data = reconcile_bill_items(pagewise_results)

    return ExtractBillDataResponse(is_success=True, data=data)
