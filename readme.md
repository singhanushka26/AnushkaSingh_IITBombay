# 🚀 Bajaj Finserv HackRx – Hybrid A+ SAFE Bill Extraction API

### **High-Accuracy Hospital Bill Parsing using Groq Vision + OCR + Multi‑Stage Refinement**

---

## 📌 **1. Overview**

This project implements a **high‑accuracy, time‑bounded (≤120s)** hospital bill extraction API built for the **Bajaj Finserv HackRx Datathon**. The system converts multi‑page PDFs or images into structured Datathon‑compliant JSON using a **multi‑stage hybrid pipeline**:

* **Scout Vision Model (Fast Pass)** to extract all pages quickly.
* **Suspicion Scoring** to detect under‑extracted or ambiguous pages.
* **Maverick Vision Model (Refinement Pass)** to reprocess ~40–70% problematic pages.
* **OCR‑Lite Row Estimation** to judge extraction completeness.
* **Two‑Stage Numeric Repair** for missing or inconsistent amounts/rates/quantities.
* **Robust JSON self‑healing** for malformed LLM output.

This pipeline is designed from scratch to **maximize accuracy while staying within Groq inference limits and real datathon constraints**.

---

## 📌 **2. End‑to‑End System Architecture**

Below is a clean, structured, ASCII‑style block diagram representing the full flow exactly as implemented in the provided `main.py`.

```
             ┌────────────────────────────┐
             │       Public Document URL   │
             └───────────────┬────────────┘
                             ▼
                ┌────────────────────────┐
                │  Download (HTTP GET)    │
                └──────────────┬──────────┘
                               ▼
                ┌────────────────────────┐
                │   PDF/Image Loader      │
                │  (Poppler / PIL Image)  │
                └──────────────┬──────────┘
                               ▼
        ┌───────────────────────────────────────────────┐
        │ Page Preprocessing                            │
        │  - smart_crop()                               │
        │  - enhance_image() (contrast + sharpness)     │
        │  - resize_image_max_dim()                     │
        │  - JPEG encode (≤4MB)                          │
        │  - OCR-lite (row estimation for suspicion)    │
        └──────────────┬─────────────────────────────────┘
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
  │  - Very sparse pages                                │
  │  - Final Bill / Pharmacy pages                      │
  │  - Last pages of PDF                                │
  │  - Zero-amount pages with OCR text                  │
  └──────────────┬──────────────────────────────────────┘
                 ▼
        ┌───────────────────────────────────┐
        │  MAVERICK REFINEMENT PASS         │
        │  - Reprocess top 40–70% pages     │
        │  - High-resolution / high-accuracy│
        └──────────────┬────────────────────┘
                       ▼
    ┌────────────────────────────────────────────┐
    │ Cleaning + JSON Reconciliation              │
    │  - Coerce numbers (empty, '-', None → 0)    │
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

## 📌 **3. Tech Stack & Components**

### **Core Technologies**

* **FastAPI** → API server
* **Pydantic v2** → strict validation & schema enforcement
* **Pillow (PIL)** → image preprocessing
* **pdf2image + Poppler** → PDF → Image conversion
* **Tesseract** → lightweight OCR for row estimation
* **Groq LPU Inference Engine** via OpenAI SDK wrapper

### **AI Models (OpenAI-compatible Groq Vision)**

* **Scout Model (Fast)**: `meta-llama/llama-4-scout-17b-16e-instruct`
* **Maverick Model (Accurate)**: `meta-llama/llama-4-maverick-17b-128e-instruct`

Each model call is handled using Groq’s `client.responses.create()` with both text + images supported.

---

## 📌 **4. Hybrid A+ SAFE Extraction Strategy**

The strategy implemented in `choose_strategy()` dynamically adapts to document size.

### **For small PDFs (≤4 pages)**

* Higher resolution
* Aggressive enhancement
* Refinement up to 3 pages

### **For medium PDFs (5–10 pages)**

* Refine ~70% of pages
* Moderate resolution

### **For large PDFs (11–20 pages)**

* Cap refinement to 12 pages
* Balanced enhancement

### **For very large PDFs (>20 pages)**

* Batch size increased
* Lower resolution
* Refinement capped tightly for speed

This adaptive strategy ensures:

* **High accuracy on dense bills**
* **Speed safety under 120 seconds**

---

## 📌 **5. Suspicion Scoring Logic**

Each page gets a weighted score based on:

### 🔹 **1. OCR rows vs extracted row mismatch**

* If OCR sees 10+ rows but only 2 items extracted → highly suspicious

### 🔹 **2. Sparse pages**

* Pages with ≤2 items + visible text

### 🔹 **3. Page types**

* `Final Bill` & `Pharmacy` are often dense → higher suspicion

### 🔹 **4. Last few pages**

* End pages frequently contain totals & missed rows

### 🔹 **5. Zero-amount pages**

* If OCR detects text → extraction likely failed

Top‑scoring pages are refined using Maverick.

---

## 📌 **6. JSON Healing & Numeric Repair Engine**

### ✔ Removes malformed structures:

* ```json fenced blocks
  ```
* stray commas
* `NaN`, `Infinity`, `-Infinity`

### ✔ Repairs numeric fields:

* Empty / dash / None → `0.0`
* If amount missing → compute `rate × qty`
* If rate missing → compute `amount ÷ qty`
* If qty missing → assume `1.0`

### ✔ Cross‑page pattern enrichment:

If many rows share the same item name:

* average rate is learned
* average qty is learned
* Missing values filled consistently

---

## 📌 **7. API Specification**

### **POST /extract-bill-data**

Input:

```json
{
  "document": "https://public-url.com/bill.pdf"
}
```

Output (Datathon format):

```json
{
  "is_success": true,
  "token_usage": { ... },
  "data": {
    "pagewise_line_items": [
      {
        "page_no": "1",
        "page_type": "Bill Detail",
        "bill_items": [
          {
            "item_name": "Accomodation Charges - ICU",
            "item_rate": 3000.0,
            "item_quantity": 2.0,
            "item_amount": 6000.0
          }
        ]
      }
    ],
    "total_item_count": 32
  }
}
```

### **GET /extract-bill-data (Health Check)**

Returns simple API status.

---

## 📌 **8. Project Structure**

```
.
├── main.py                  # Full Hybrid A+ SAFE pipeline
├── Dockerfile               # For container deployment
├── Procfile                 # For Railway deployment
├── requirements.txt         # Python dependencies
├── readme.md                # (this document)
└── .gitignore
```

---

## 📌 **9. Key Strengths of My Implementation**

* Purpose-built for HackRx datathon constraints
* High accuracy with dual-model hybrid flow
* Time-safe (sub‑120 seconds)
* JSON self-healing prevents API failures
* Uses OCR only for structural cues (fast)
* Automatic numeric consistency engine
* Strong fault tolerance for noisy PDFs
* Adaptive strategy based on page count

---

## 📌 **10. Future Enhancements**

* Integration of bounding-box extraction
* Table detection using segmentation models
* Parallel refinement calls for speed
* Noise-aware OCR boosting
* Option for Extreme Accuracy Mode (full Maverick)

---

## 🎉 Final Notes

This README represents the architecture exactly as implemented in the provided `main.py`. All core components, design choices, algorithms, and hybrid refinements are documented precisely and cleanly.
