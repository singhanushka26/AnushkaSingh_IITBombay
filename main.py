import base64
import io
import json
import math
import mimetypes
import os
import time
import asyncio
from typing import List, Optional, Tuple, Any, Dict
from collections import defaultdict

import requests
import cv2  # OpenCV for the "Differentiator" (Preprocessing)
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from pdf2image import convert_from_bytes
from PIL import Image, ImageEnhance
from openai import AsyncOpenAI

# ============================================================
#   Configuration
# ============================================================

# Initialize Async Client for parallel processing
client = AsyncOpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

# Using the standard Llama 3.2 Vision model (High speed/Good accuracy)
GROQ_VISION_MODEL_ID = os.environ.get(
    "GROQ_VISION_MODEL_ID",
    "llama-3.2-11b-vision-preview" 
)

# STRICT CONSTRAINT: Process exactly 5 pages in parallel.
# This respects your "cannot handle more than 5 pages in one go" rule
# while ensuring 5x speedup over sequential processing.
MAX_CONCURRENT_REQUESTS = 5

# ============================================================
#   Pydantic Schemas (Exact Datathon Spec)
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

app = FastAPI(
    title="Bajaj Datathon Bill Extraction API",
    description="High-accuracy extraction using Bounded Async Parallelism (Max 5)."
)

# ============================================================
#   Image Processing (The Differentiator)
# ============================================================

def preprocess_image_opencv(pil_image: Image.Image) -> Image.Image:
    """
    DIFFERENTIATOR: Uses Adaptive Thresholding to handle scanned photos,
    shadows, and uneven lighting. This dramatically improves OCR accuracy.
    """
    # Convert PIL to OpenCV format
    open_cv_image = np.array(pil_image)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    
    # 1. Grayscale
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    
    # 2. Adaptive Thresholding
    # This calculates the threshold for small regions, perfect for shadows
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    return Image.fromarray(binary)

def image_to_data_url(img: Image.Image, max_dim: int = 1600) -> str:
    """
    Resizes image and converts to base64.
    Increased max_dim to 1600 because medical bills have small decimals.
    """
    w, h = img.size
    scale = max(w, h) / float(max_dim)
    
    if scale > 1.0:
        new_w = int(w / scale)
        new_h = int(h / scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    
    # Apply the Differentiator Preprocessing
    try:
        img = preprocess_image_opencv(img)
    except Exception as e:
        print(f"OpenCV processing failed, falling back to PIL: {e}")
        img = ImageEnhance.Contrast(img).enhance(1.5)
        img = ImageEnhance.Sharpness(img).enhance(1.5)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def download_document(url: str) -> bytes:
    try:
        resp = requests.get(url, timeout=40)
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def document_to_page_infos(url: str, content: bytes) -> List[Dict[str, Any]]:
    # Guess Mime
    mime, _ = mimetypes.guess_type(url)
    if not mime and content[:4] == b"%PDF":
        mime = "application/pdf"
    
    pil_images = []
    
    if mime == "application/pdf":
        try:
            # Requires poppler-utils installed on system
            pil_images = convert_from_bytes(content)
        except Exception as e:
             raise HTTPException(status_code=400, detail=f"PDF Error: {e}")
    else:
        try:
            pil_images = [Image.open(io.BytesIO(content)).convert("RGB")]
        except Exception as e:
             raise HTTPException(status_code=400, detail=f"Image Error: {e}")

    # Return list of dicts to be processed
    return [{"page_index": i, "pil_image": img} for i, img in enumerate(pil_images)]

# ============================================================
#   Prompts & Async Logic
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
    Process a single page. The semaphore limits us to MAX_CONCURRENT_REQUESTS (5).
    """
    page_idx = page_info["page_index"]
    pil_img = page_info["pil_image"]
    
    # Preprocess & Encode
    b64_url = image_to_data_url(pil_img)
    
    async with semaphore:  # This line enforces the "Max 5" limit
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
                temperature=0.1,  # Low temp for accuracy
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
                # Handle model potentially wrapping output in extra keys
                if "pagewise_line_items" in data:
                     data = data["pagewise_line_items"][0]
                
                # Ensure page_no is set correctly
                data["page_no"] = str(page_idx + 1)
                return data, token_usage
                
            except json.JSONDecodeError:
                print(f"JSON Decode Error on Page {page_idx+1}")
                return {
                    "page_no": str(page_idx + 1),
                    "page_type": "Error",
                    "bill_items": []
                }, token_usage

        except Exception as e:
            print(f"API Error on Page {page_idx+1}: {e}")
            # Return empty structure on failure so other pages still succeed
            return {
                "page_no": str(page_idx + 1),
                "page_type": "Error",
                "bill_items": []
            }, TokenUsage(total_tokens=0, input_tokens=0, output_tokens=0)

# ============================================================
#   Helpers: Cleaning & Pattern Enrichment
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
    """Clean and validate raw JSON from LLM"""
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
    """
    Improves accuracy by using patterns from clear rows to fix unclear rows.
    (e.g., if one 'Consultation' row has Price=1000, assume others do too)
    """
    rates = defaultdict(list)
    qtys = defaultdict(list)

    # Pass 1: Collect patterns
    for p in pages:
        for it in p.bill_items:
            name_key = it.item_name.strip().lower()
            if it.item_rate > 0: rates[name_key].append(it.item_rate)
            if it.item_quantity > 0: qtys[name_key].append(it.item_quantity)
            elif it.item_amount > 0 and it.item_rate > 0:
                 qtys[name_key].append(it.item_amount / it.item_rate)

    avg_rate = {k: sum(v)/len(v) for k, v in rates.items() if v}
    avg_qty = {k: sum(v)/len(v) for k, v in qtys.items() if v}

    # Pass 2: Fill gaps
    for p in pages:
        for it in p.bill_items:
            name_key = it.item_name.strip().lower()
            
            # If Qty missing, infer from average
            if it.item_quantity <= 0 and name_key in avg_qty:
                it.item_quantity = avg_qty[name_key]
            
            # If Rate missing, infer from average
            if it.item_rate <= 0 and name_key in avg_rate:
                it.item_rate = avg_rate[name_key]
            
            # Recalculate Amount if missing but Rate/Qty exist
            if it.item_amount <= 0 and it.item_rate > 0 and it.item_quantity > 0:
                it.item_amount = it.item_rate * it.item_quantity
                
    return pages

def aggregate_all_pages(pages: List[PageItems]) -> ExtractBillDataResponseData:
    total_items = sum(len(p.bill_items) for p in pages)
    # Sort by page number to be nice
    pages.sort(key=lambda x: int(x.page_no) if x.page_no.isdigit() else 0)
    return ExtractBillDataResponseData(pagewise_line_items=pages, total_item_count=total_items)

# ============================================================
#   Main Endpoint
# ============================================================

@app.post("/extract-bill-data", response_model=ExtractBillDataResponse)
async def extract_bill_data(req: ExtractBillDataRequest):
    start_time = time.time()
    url_str = str(req.document)

    # 1. Download & Convert
    try:
        content = download_document(url_str)
        raw_pages = document_to_page_infos(url_str, content)
    except Exception as e:
        return ExtractBillDataResponse(is_success=False, message=str(e))

    if not raw_pages:
        return ExtractBillDataResponse(is_success=False, message="No pages found.")

    # 2. Async Parallel Processing (Bounded by MAX_CONCURRENT_REQUESTS = 5)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = [process_single_page(p, semaphore) for p in raw_pages]
    
    # Run all tasks (but only 5 active at a time)
    results = await asyncio.gather(*tasks)

    # 3. Aggregate Results
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

    # 4. Post-Processing & Pattern Enrichment
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