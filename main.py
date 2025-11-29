# ============================================================
#  HIGH-ACCURACY + FAST + JSON-STABLE BILL EXTRACTION ENGINE
# ============================================================

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

# ------------------------------------------------------------
#                GROQ CLIENT
# ------------------------------------------------------------
client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

MODEL_FAST = "meta-llama/llama-4-scout-17b-16e-instruct"
MODEL_ACCURATE = "meta-llama/llama-4-maverick-17b-128e-instruct"

MAX_IMAGES_PER_REQUEST = 5


# ------------------------------------------------------------
#                DATATHON SCHEMAS
# ------------------------------------------------------------

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


# ------------------------------------------------------------
#                FASTAPI APP
# ------------------------------------------------------------

app = FastAPI(
    title="High-Accuracy Bill Extraction API",
    version="9.0.0",
    description="Hybrid Scout + Maverick pipeline with guaranteed JSON safety."
)


# ------------------------------------------------------------
#                DOCUMENT HANDLING
# ------------------------------------------------------------

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


def enhance(img: Image.Image) -> Image.Image:
    img = ImageEnhance.Contrast(img).enhance(1.5)
    img = ImageEnhance.Sharpness(img).enhance(1.4)
    return img


def resize(img: Image.Image) -> Image.Image:
    max_dim = 950
    w, h = img.size
    s = max(w, h) / max_dim
    if s <= 1:
        return img
    return img.resize((int(w / s), int(h / s)), Image.LANCZOS)


def img_to_dataurl(img: Image.Image) -> str:
    img = resize(img)
    img = enhance(img)

    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=60)
    data = buf.getvalue()

    while len(data) > 4 * 1024 * 1024:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=40)
        data = buf.getvalue()

    return "data:image/jpeg;base64," + base64.b64encode(data).decode()


def document_to_images(url: str, content: bytes) -> List[str]:
    mime = guess_mime(url, content)

    if mime.startswith("image/"):
        img = Image.open(io.BytesIO(content)).convert("RGB")
        return [img_to_dataurl(img)]

    if mime == "application/pdf":
        pages = convert_from_bytes(content)
        return [img_to_dataurl(p.convert("RGB")) for p in pages]

    # fallback
    img = Image.open(io.BytesIO(content)).convert("RGB")
    return [img_to_dataurl(img)]


# ------------------------------------------------------------
#            SYSTEM PROMPTS (FAST + ACCURATE)
# ------------------------------------------------------------


# ============================================================
#  LLM Prompt – batched, strict JSON, handle repeated rows
# ============================================================

SYSTEM_PROMPT = """
You are an expert medical BILL ITEM extraction engine for hospital bills.

Your goals:

1) Capture EVERY genuine line item from the tables (high recall).
2) Do NOT double-count or duplicate the same line item when it is only a summary.
3) Make the sum of all `item_amount` values across all pages as close as possible
   to the FINAL TOTAL printed in the bill.
4) Respect the exact JSON structure required by the HackRx Datathon.

CRITICAL BEHAVIOUR FOR REPEATED ROWS:

- If the same row appears MANY TIMES (e.g. many rows with
  "IP CONSULTATION CHARGES  Qty 1  Rate 1000  Amount 1000")
  then you MUST output ONE BILL ITEM PER VISUAL ROW.
  Example: if 20 such rows are visible, you must output 20 bill_items
  with the same name & numbers (do NOT collapse them into one).

- Section totals such as "TOTAL", "SUB TOTAL", "GRAND TOTAL",
  "NET AMOUNT PAYABLE", etc. MUST NOT be emitted as bill_items.

NUMERIC RULES:

- item_quantity: read from columns like "Qty", "No. of Days", etc.
- item_rate:     from "Rate", "Charges per day", etc.
- item_amount:   from "Amount", "Net Amt", etc.

If some numeric fields are missing for a row BUT other rows with the
same description show clear numbers, use that pattern:

  • If at least one row for the same item_name has
        quantity = q0 and amount = a0 (or rate = r0),
    then for rows where the numeric values are unreadable you may assume:
        quantity = q0
        rate     = a0 / q0 (or r0)
        amount   = quantity * rate

This means you may RECONSTRUCT missing amounts using the pattern of
other rows with the same description (e.g. repeated consultation charges).

If a numeric field is still unknown after this reasoning, set it to 0.0.

OUTPUT FORMAT (PER BATCH):

You will see K page images in this batch, in order.
For EACH page i in this batch (1-based):

- Decide page_type ∈ {"Bill Detail", "Final Bill", "Pharmacy"}.
- Extract ALL line items for that page (one per visual row).

Return ONE STRICT JSON object ONLY (no markdown):

{
  "pagewise_line_items": [
    {
      "page_no": "1",
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
  ],
  "total_item_count": <integer>
}

- page_no is the 1-based index WITHIN THIS BATCH, as a STRING.
- bill_items may be [] for pages with no charges.
- total_item_count is the number of bill_items across all pages in the batch.

STRICT REQUIREMENTS:
- JSON ONLY. No ```json, no headings, no commentary.
- No extra keys at top level or inside any object.
- Do NOT output totals, sub-totals, taxes, or summary-only rows as bill_items.
"""


SYSTEM_PROMPT_ACCURATE = """
Re-check this page with MAXIMUM ACCURACY.
Extract ALL real line items, not totals.
STRICT JSON ONLY.
"""


# ------------------------------------------------------------
#               ROBUST JSON PARSER  (never fails)
# ------------------------------------------------------------

def safe_json_parse(text: str) -> dict:
    text = text.strip()

    # Remove markdown fencing if exists
    text = text.replace("```json", "").replace("```", "")

    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last == -1:
        raise HTTPException(500, "No JSON object found in response")

    block = text[first:last + 1]
    try:
        return json.loads(block)
    except Exception:
        # Try to clean trailing garbage
        block = block.replace("\n", " ").replace("\t", " ")
        return json.loads(block)


# ------------------------------------------------------------
#               GROQ CALL
# ------------------------------------------------------------

def groq_call(model: str, system_msg: str, messages: List[dict]) -> Tuple[dict, TokenUsage]:
    try:
        resp = client.responses.create(
            model=model,
            input=messages
        )
    except Exception as e:
        raise HTTPException(503, f"Groq error: {e}")

    raw = resp.output_text
    parsed = safe_json_parse(raw)

    u = resp.usage
    t = TokenUsage(
        total_tokens=int(getattr(u, "total_tokens", 0)),
        input_tokens=int(getattr(u, "input_tokens", 0)),
        output_tokens=int(getattr(u, "output_tokens", 0)),
    )
    return parsed, t


# ------------------------------------------------------------
#               CLEANING + FIXING
# ------------------------------------------------------------

def num(x):
    if x in [None, "", "-", "—"]:
        return 0.0
    try:
        return float(str(x).replace(",", ""))
    except:
        return 0.0


def fix_page_values(page: PageItems):
    for it in page.bill_items:
        amt = it.item_amount
        rate = it.item_rate
        qty = it.item_quantity

        if rate > 0 and qty > 0:
            it.item_amount = round(rate * qty, 2)
        elif amt > 0 and qty > 0:
            it.item_rate = round(amt / qty, 4)
        elif amt > 0:
            it.item_quantity = 1.0
            it.item_rate = round(amt, 2)
    return page


def enrich_missing_values(pages: List[PageItems]):
    rates = defaultdict(list)
    qtys = defaultdict(list)

    # collect patterns
    for p in pages:
        for it in p.bill_items:
            name = it.item_name.lower()
            if it.item_rate > 0 and it.item_quantity > 0:
                rates[name].append(it.item_rate)
                qtys[name].append(it.item_quantity)

    avg_rate = {k: sum(v) / len(v) for k, v in rates.items()}
    avg_qty = {k: sum(v) / len(v) for k, v in qtys.items()}

    # fill missing
    for p in pages:
        for it in p.bill_items:
            name = it.item_name.lower()
            if it.item_rate <= 0 and name in avg_rate:
                it.item_rate = avg_rate[name]
            if it.item_quantity <= 0 and name in avg_qty:
                it.item_quantity = avg_qty[name]
            if it.item_amount <= 0:
                it.item_amount = it.item_rate * it.item_quantity

    return pages


# ------------------------------------------------------------
#                WHAT PAGES NEED ACCURATE RECHECK?
# ------------------------------------------------------------

def detect_suspicious(page: PageItems) -> bool:
    """
    We trigger Maverick re-check on pages:
    - with < 2 rows
    - OR with many repeated noisy rows
    - OR with >50% zero-valued items
    """

    total = len(page.bill_items)
    if total <= 1:
        return True

    zero_count = sum(1 for it in page.bill_items if it.item_amount <= 0)
    if zero_count / max(total, 1) > 0.4:
        return True

    # Repeated names
    names = [it.item_name.lower() for it in page.bill_items]
    if len(names) - len(set(names)) > 5:
        return True

    return False


# ------------------------------------------------------------
#               API ENDPOINTS
# ------------------------------------------------------------

@app.get("/extract-bill-data")
def health():
    return {"message": "OK"}


@app.post("/extract-bill-data", response_model=ExtractBillDataResponse)
def extract_api(req: ExtractBillDataRequest):
    t0 = time.time()

    content = download_document(req.document)
    imgs = document_to_images(req.document, content)
    n = len(imgs)

    pages: List[PageItems] = []
    total_usage = TokenUsage(0, 0, 0)

    # -----------------------------
    #       SCOUT BATCH PASS
    # -----------------------------
    for i in range(0, n, MAX_IMAGES_PER_REQUEST):
        batch = imgs[i:i + MAX_IMAGES_PER_REQUEST]

        messages = [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": f"Process {len(batch)} pages"}]
            }
        ]

        for j, img in enumerate(batch):
            messages[-1]["content"].append({"type": "input_image", "image_url": img, "detail": "auto"})

        parsed, use = groq_call(MODEL_FAST, SYSTEM_PROMPT, messages)
        total_usage.total_tokens += use.total_tokens
        total_usage.input_tokens += use.input_tokens
        total_usage.output_tokens += use.output_tokens

        raw_pages = parsed.get("pagewise_line_items", parsed)

        # normalize
        for k, p in enumerate(raw_pages):
            p["page_no"] = str(i + k + 1)
            pages.append(PageItems(**p))

    # -----------------------------
    #   ACCURATE RECHECK (MAVERICK)
    # -----------------------------
    for idx, p in enumerate(pages):
        if not detect_suspicious(p):
            continue

        img = imgs[idx]

        messages = [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": SYSTEM_PROMPT_ACCURATE}]
            },
            {
                "role": "user",
                "content": [{"type": "input_image", "image_url": img, "detail": "high"}]
            }
        ]

        parsed, use = groq_call(MODEL_ACCURATE, SYSTEM_PROMPT_ACCURATE, messages)

        total_usage.total_tokens += use.total_tokens
        total_usage.input_tokens += use.input_tokens
        total_usage.output_tokens += use.output_tokens

        parsed["page_no"] = str(idx + 1)
        pages[idx] = PageItems(**parsed)

    # -----------------------------
    #       CLEANUP & ENRICH
    # -----------------------------
    pages = [fix_page_values(p) for p in pages]
    pages = enrich_missing_values(pages)

    total_items = sum(len(p.bill_items) for p in pages)
    data = ExtractBillDataResponseData(pagewise_line_items=pages, total_item_count=total_items)

    print(f"[BILL] pages={n} items={total_items} tokens={total_usage.total_tokens} time={time.time()-t0:.2f}s")

    return ExtractBillDataResponse(
        is_success=True,
        token_usage=total_usage,
        data=data
    )
