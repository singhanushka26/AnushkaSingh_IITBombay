# Invoice Line Item Extractor (Groq + Llama 4 Maverick)

## Problem Statement
Given invoice/bill documents (images or PDFs with multiple pages),
extract all line items and compute:
- per-line `item_amount`, `item_rate`, `item_quantity`
- `total_item_count`
- `reconciled_amount` (sum of all line items, no double counting)

## Tech Stack
- FastAPI (Python backend)
- Groq Responses API
- Model: `meta-llama/llama-4-maverick-17b-128e-instruct`
- pdf2image + Pillow for PDF-to-image conversion

## API Specification

### Endpoint
`POST /extract-bill-data`

### Request Body
```json
{
  "document": "<public-url-to-image-or-pdf>"
}
