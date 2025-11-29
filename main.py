import base64
import io
import json
import os
import time
import asyncio
import mimetypes
from collections import defaultdict
from typing import List, Optional, Tuple, Any, Dict

import requests
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl

# Replaces OpenCV for Railway compatibility
from PIL import Image, ImageEnhance, ImageOps 
from pdf2image import convert_from_bytes
from openai import AsyncOpenAI

# ============================================================
#   Configuration
# ============================================================

client = AsyncOpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

# Llama 3.2 11B Vision: Best balance of speed/accuracy for Datathons
GROQ_VISION_MODEL_ID = os.environ.get(
    "GROQ_VISION_MODEL_ID",
    "llama-3.2-11b-vision-preview"
)

# STRICT CONSTRAINT: Max 5 concurrent requests to respect rate limits
# and "5 pages in one go" rule.
MAX_CONCURRENT_REQUESTS = 5

# ============================================================
#   Pydantic Models
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
#   FastAPI App
# ============================================================

app = FastAPI(title="Bajaj Datathon API - Railway Compatible")

# ============================================================
#   Image Processing (Pure PIL - No OpenCV)
# ============================================================

def preprocess_image_pil(img: Image.Image) -> Image.Image:
    """
    Railway-safe preprocessing. 
    Uses PIL instead of OpenCV to remove shadows and enhance text.
    """
    # 1. Convert to Grayscale
    img = img.convert("L")
    
    # 2. Auto-Contrast (Cutoff 1% of pixels) - removes grey/shadow background
    img = ImageOps.autocontrast(img, cutoff=1)
    
    # 3. Increase Sharpness (Helps OCR read edges)
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(2.0)  # Double sharpness
    
    # 4. Slight Contrast Boost
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)
    
    # Convert back to RGB for the model
    return img.convert("RGB")

def image_to_data_url(img: Image.Image, max_dim: int = 1600) -> str:
    """
    Resize and encode. max_dim=1600 ensures decimals are readable.
    """
    w, h = img.size
    scale = max(w, h) / float(max_dim)
    
    if scale > 1.0:
        new_w = int(w / scale)
        new_h = int(h / scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    
    # Apply Railway-safe preprocessing
    img = preprocess_image_pil(img)

    buf = io.BytesIO()
    # High quality JPEG
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def download_document(url: str) -> bytes:
    try:
        resp = requests.get(url, timeout=40)
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Download failed: {e}")

def document_to_page_infos(url: str, content: bytes) -> List[Dict[str, Any]]:
    mime, _ = mimetypes.guess_type(url)
    if not mime and content[:4] == b"%PDF":
        mime = "application/pdf"
    
    pil_images = []
    
    if mime == "application/pdf":
        try:
            # Note: On Railway, you may need to add 'poppler-utils' via Nixpacks 
            # if this fails. But usually pdf2image handles bytes well.
            pil_images = convert_from_bytes(content)
        except Exception as e:
             raise HTTPException(
                 status_code=400, 
                 detail=f"PDF Conversion Error (Ensure poppler is installed): {e}"
             )
    else:
        try:
            pil_images = [Image.open(io.BytesIO(content)).convert("RGB")]
        except Exception as e:
             raise HTTPException(status_code=400, detail=f"Image Error: {e}")

    return [{"page_index": i, "pil_image": img} for i, img in enumerate(pil_images)]

# ============================================================
#   Prompt & Async Logic
# ============================================================

SYSTEM_PROMPT = """
You are a precision OCR engine for medical bills.
Extract the line items from the provided image into strict JSON.

RULES:
1. Extract items EXACTLY as printed.
2. If you see multiple rows for "Consultation", output multiple items. Do not merge them.
3. Be careful with decimal points in Amounts (100.00 vs 10000).
4. Ignore "Sub Total", "Total", "Discount", "Tax" lines. Extract only the individual line items.

JSON FORMAT:
{
  "page_type": "Bill Detail" | "Final Bill" | "Pharmacy",
  "bill_items": [
    {
      "item_name": "string",
      "item_amount": float, 
      "item_rate": float, 
      "item_quantity": float
    }
  ]
}
If Rate or Quantity are missing/blank, set to 0.0.
"""

async def process_single_page(
    page_info: Dict[str, Any], 
    semaphore: asyncio.Semaphore
) -> Tuple[Dict[str, Any], TokenUsage]:
    """
    Process one page with concurrency limit.
    """
    page_idx = page_info["page_index"]
    pil_img = page_info["pil_image"]
    
    # Resize & Enhance (Pure Python)
    b64_url = image_to_data_url(pil_img)
    
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=GROQ_VISION_MODEL_ID,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract line items from this page."},
                            {
                                "type": "image_url",
                                "image_url": {"url": b64_url}
                            }
                        ]
                    }
                ],
                temperature=0.1,
                max_tokens=2048,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            usage = response.usage
            token_usage = TokenUsage(
                total_tokens=usage.total_tokens,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens
            )
            
            try:
                data = json.loads(content)
                if "pagewise_line_items" in data:
                     data = data["pagewise_line_items"][0]
                
                data["page_no"] = str(page_idx + 1)
                return data, token_usage
            except json.JSONDecodeError:
                # Return empty page on JSON error so entire process doesn't fail
                return {
                    "page_no": str(page_idx + 1),
                    "page_type": "Error",
                    "bill_items": []
                }, token_usage

        except Exception as e:
            print(f"Page {page_idx+1} failed: {e}")
            return {
                "page_no": str(page_idx + 1),
                "page_type": "Error",
                "bill_items": []
            }, TokenUsage(total_tokens=0, input_tokens=0, output_tokens=0)

# ============================================================
#   Data Cleaning & Aggregation
# ============================================================

def _coerce_number(x: Any) -> float:
    if x is None: return 0.0
    if isinstance(x, (int, float)): return float(x)
    if isinstance(x, str):
        s = x.strip().replace(",", "")
        if s == "" or s in {"-", "—", "NA", "N/A"}: return 0.0
        try: return float(s)
        except: return 0.0
    return 0.0

def reconcile_page_items(page_dict: Dict[str, Any]) -> PageItems:
    raw_items = page_dict.get("bill_items", []) or []
    cleaned_items = []
    
    for item in raw_items:
        if not isinstance(item, dict): continue
        cleaned_items.append({
            "item_name": str(item.get("item_name", "")).strip(),
            "item_amount": _coerce_number(item.get("item_amount")),
            "item_rate": _coerce_number(item.get("item_rate")),
            "item_quantity": _coerce_number(item.get("item_quantity")),
        })
        
    return PageItems(
        page_no=str(page_dict.get("page_no", "")),
        page_type=str(page_dict.get("page_type", "Bill Detail")),
        bill_items=cleaned_items
    )

def enrich_from_patterns(pages: List[PageItems]) -> List[PageItems]:
    """Pattern matching to fill missing rates/quantities"""
    rates = defaultdict(list)
    qtys = defaultdict(list)

    # Learn patterns
    for p in pages:
        for it in p.bill_items:
            name_key = it.item_name.strip().lower()
            if it.item_rate > 0: rates[name_key].append(it.item_rate)
            if it.item_quantity > 0: qtys[name_key].append(it.item_quantity)
            elif it.item_amount > 0 and it.item_rate > 0:
                 qtys[name_key].append(it.item_amount / it.item_rate)

    avg_rate = {k: sum(v)/len(v) for k, v in rates.items() if v}
    avg_qty = {k: sum(v)/len(v) for k, v in qtys.items() if v}

    # Apply patterns
    for p in pages:
        for it in p.bill_items:
            name_key = it.item_name.strip().lower()
            if it.item_quantity <= 0 and name_key in avg_qty:
                it.item_quantity = avg_qty[name_key]
            if it.item_rate <= 0 and name_key in avg_rate:
                it.item_rate = avg_rate[name_key]
            if it.item_amount <= 0 and it.item_rate > 0 and it.item_quantity > 0:
                it.item_amount = it.item_rate * it.item_quantity
                
    return pages

def aggregate_all_pages(pages: List[PageItems]) -> ExtractBillDataResponseData:
    total_items = sum(len(p.bill_items) for p in pages)
    pages.sort(key=lambda x: int(x.page_no) if x.page_no.isdigit() else 0)
    return ExtractBillDataResponseData(pagewise_line_items=pages, total_item_count=total_items)

# ============================================================
#   Main Endpoint
# ============================================================

@app.get("/")
def home():
    return {"status": "Active", "model": GROQ_VISION_MODEL_ID}

@app.post("/extract-bill-data", response_model=ExtractBillDataResponse)
async def extract_bill_data(req: ExtractBillDataRequest):
    start_time = time.time()
    url_str = str(req.document)

    try:
        content = download_document(url_str)
        raw_pages = document_to_page_infos(url_str, content)
    except Exception as e:
        return ExtractBillDataResponse(is_success=False, message=str(e))

    if not raw_pages:
        return ExtractBillDataResponse(is_success=False, message="No pages found.")

    # Async Processing with Semaphore
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = [process_single_page(p, semaphore) for p in raw_pages]
    
    results = await asyncio.gather(*tasks)

    # Aggregation
    all_pages_data = []
    total_tokens = 0
    input_tokens = 0
    output_tokens = 0

    for res_data, res_usage in results:
        cleaned_page = reconcile_page_items(res_data)
        all_pages_data.append(cleaned_page)
        
        total_tokens += res_usage.total_tokens
        input_tokens += res_usage.input_tokens
        output_tokens += res_usage.output_tokens

    # Post-processing
    if all_pages_data:
        all_pages_data = enrich_from_patterns(all_pages_data)

    data_out = aggregate_all_pages(all_pages_data)
    
    elapsed = time.time() - start_time
    print(f"Processed {len(raw_pages)} pages in {elapsed:.2f}s. Items: {data_out.total_item_count}")

    return ExtractBillDataResponse(
        is_success=True,
        token_usage=TokenUsage(
            total_tokens=total_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        ),
        data=data_out
    )