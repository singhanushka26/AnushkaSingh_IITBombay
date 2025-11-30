# Bajaj Finserv Datathon – Hybrid Vision + OCR Bill Extraction API

### **High-Accuracy Hospital Bill Parsing using Groq Vision Models + OCR + Multi-Stage Refinement**

---

##   **1. Project Overview**

This project implements a **high-accuracy** hospital bill extraction API built specifically for the **Bajaj Finserv Datathon**. It converts multi-page hospital bills (PDFs or images) into clean, structured JSON following the exact competition schema.

The system uses a **hybrid multi-stage pipeline** combining:

* **Groq Vision – Scout Model** for bulk extraction (fast)
* **Groq Vision – Maverick Model** for selective refinement (accurate)
* **OCR-Lite row estimation** to detect missing line items
* **Robust image preprocessing pipeline** (crop + enhance + resize)
* **Automatic numeric repair engine** for fixing missing/inconsistent values
* **Global pattern enrichment** to fill rates/qty using learned statistics
* **Fully self-healing JSON parser** to recover from malformed LLM output

The entire pipeline is optimized to deliver high accuracy while remaining robust across noisy scans, multi-table layouts, and large multi-page documents.

---

##   **2. End-to-End Architecture Diagram**

```
             ┌────────────────────────────┐
             │       Public Document URL  │
             └───────────────┬────────────┘
                             ▼
                ┌─────────────────────────┐
                │  Download (HTTP GET)    │
                └──────────────┬──────────┘
                               ▼
                ┌────────────────────────┐
                │   PDF/Image Loader      │
                │   (pdf2image / PIL)     │
                └──────────────┬──────────┘
                               ▼
        ┌───────────────────────────────────────────────┐
        │ Page Preprocessing                            │
        │  - smart_crop()                               │
        │  - enhance_image() (contrast + sharpness)     │
        │  - resize_image_max_dim()                     │
        │  - JPEG encode (≤4MB)                         │
        │  - OCR-Lite (row estimation)                  │
        └──────────────┬────────────────────────────────┘
                       ▼
          ┌─────────────────────────────┐
          │  BULK PASS (Scout Vision)   │
          │  - Batch size = 2–4         │
          │  - Extract ALL pages        │
          └──────────────┬──────────────┘
                         ▼
  ┌─────────────────────────────────────────────────────┐
  │ Suspicion Analyzer                                  │
  │  - OCR rows vs extracted rows                       │
  │  - Sparse pages                                     │
  │  - Final Bill / Pharmacy pages                      │
  │  - Last pages                                       │
  │  - Zero-amount pages with OCR text                  │
  └──────────────┬──────────────────────────────────────┘
                 ▼
        ┌───────────────────────────────────┐
        │  MAVERICK REFINEMENT PASS         │
        │  - Reprocess top 40–70% pages     │
        │  - High-resolution, high-accuracy │
        └──────────────┬────────────────────┘
                       ▼
    ┌─────────────────────────────────────────────┐
    │ Cleaning + JSON Reconciliation              │
    │  - Coerce numbers                           │
    │  - Infer qty/rate when missing              │
    │  - amount = rate × qty correction           │
    └──────────────┬──────────────────────────────┘
                   ▼
  ┌──────────────────────────────────────────────────┐
  │ Global Pattern Enrichment                        │
  │  - Learn typical rate per item_name              │
  │  - Learn typical qty per item_name               │
  │  - Fill broken / inconsistent rows               │
  └──────────────┬───────────────────────────────────┘
                 ▼
  ┌──────────────────────────────────────────────────┐
  │ Final Aggregation                                │
  │  - pagewise_line_items                           │
  │  - total_item_count                              │
  └──────────────┬───────────────────────────────────┘
                 ▼
        ┌─────────────────────────────┐
        │   Final JSON Response       │
        │ (Strict Datathon Schema)    │
        └─────────────────────────────┘
```

---

##   **3. Technology Stack**

### **AI Models (Groq LPU Inference)**

* **Fast Extraction:** `meta-llama/llama-4-scout-17b-16e-instruct`
* **Accurate Refinement:** `meta-llama/llama-4-maverick-17b-128e-instruct`

### **Backend**

* FastAPI
* Pydantic v2
* Python 3.10+

### **Vision & Processing**

* Pillow (PIL)
* pdf2image (uses pdf2image’s internal engine)
* Tesseract OCR for row estimation

### **Networking**

* Requests (for downloading public URLs)

---

## **4. Running Locally**
### **Install Dependencies**
```
pip install -r requirements.txt
```

### **Environment Variable**
```
export GROQ_API_KEY="your-key"
```

### **Run the Server**
```
uvicorn main:app --host 0.0.0.0 --port 8000
```

### **Test**
```
curl -X POST http://localhost:8000/extract-bill-data \
  -H "Content-Type: application/json" \
  -d '{"document": "https://example.com/bill.pdf"}'
```

---

## **5. Deployment**
Works on:
- Railway
- Render
- AWS EC2
- Docker

Your repo already contains:
- `Dockerfile`
- `Procfile`
- `requirements.txt`
- `main.py`

---

##   **6. Pipeline Logic**

### **1. Bulk Extraction (Scout)**

* Processes all pages quickly
* Medium-high resolution
* Extracts initial line items
* Minimal latency

### **2. Suspicion Scoring**

Each page is scored using:

* OCR rows vs extracted items mismatch
* Sparse bill items
* Page type (Final Bill, Pharmacy)
* Last pages of PDF
* Pages with zero amount but visible OCR text

### **3. Refinement (Maverick)**

* Re-extract ~40–70% most suspicious pages
* Much higher accuracy
* Better structure recognition
* Fixes missed rows, wrong splits, numeric misreads

### **4. Cleaning & Reconciliation**

The system actively repairs:

* Missing numeric values
* Incorrect amount vs rate×qty
* Empty fields (`''`, None, '-', '—')
* Schema normalization

### **5. Pattern Enrichment**

Learns global patterns across all pages:

* Most frequent rate for each item name
* Most frequent quantity
* Fills missing data consistently

### **6. Final Aggregation**

Produces:

* `pagewise_line_items` (per page extraction)
* `total_item_count`

---

## **7. API Routes**

### **GET /extract-bill-data**

Returns API health message.

### **POST /extract-bill-data**

Request:

```json
{
  "document": "https://public-url.com/bill.pdf"
}
```

Response:

```json
{
        "is_success": "boolean", // If Status code 200 and following valid schema, then true
        "token_usage": {
            "total_tokens": "integer", // Cumulative Tokens from all LLM calls
            "input_tokens": "integer", // Cumulative Tokens from all LLM calls
            "output_tokens": "integer" // Cumulative Tokens from all LLM calls
        },
        "data": {
            "pagewise_line_items": [
            {
                "page_no": "string",
                "page_type": "Bill Detail | Final Bill | Pharmacy",
                "bill_items": [
                {
                    "item_name": "string", // Exactly as mentioned in the bill
                    "item_amount": "float", // Net Amount of the item post discounts as mentioned in the bill
                    "item_rate": "float", // Exactly as mentioned in the bill
                    "item_quantity": "float" // Exactly as mentioned in the bill
                }
                ]
            }
            ],
            "total_item_count": "integer" // Count of items across all pages
        }
}
```

---

## **8. Project Structure**

```
.
├── main.py
├── Dockerfile
├── Procfile
├── requirements.txt
├── readme.md
└── .gitignore
```

---

## **9. Why This Pipeline Works Well**

* Dual-stage extraction: fast + accurate
* OCR used only for structural cues (fast, not heavy OCR)
* Handles noisy scans, shadows, multiple tables
* JSON self-healing prevents model failures
* Global patterns ensure consistent numeric reconstruction
* Adaptive strategy based on number of pages

---

## **10. Future Enhancements**

* Bounding-box extraction
* Table segmentation model
* Full-page semantic segmentation
* Parallel refinement calls
* Faster image downscaling & compression


---

## **11. Contact**
* Name:     Anushka
* Email:    22b0714@iitb.ac.in
* College:  IIT Bombay
---

## Final Notes

This README represents the architecture exactly as implemented in the provided `main.py`. All core components, design choices, algorithms, and hybrid refinements are documented precisely and cleanly.
