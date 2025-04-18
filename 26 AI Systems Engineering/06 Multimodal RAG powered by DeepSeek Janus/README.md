# Project 6: Multimodal RAG powered by DeepSeek Janus

## Overview
You’ll build a RAG system that:

Ingests text and image documents

Stores them with associated embeddings

At query time, retrieves relevant content (text or image)

Feeds the results to DeepSeek-VL (Janus) for a multimodal response

## Tech Stack

### ComponentTool

LibraryMultimodal LLM[DeepSeek-VL via Ollama or local API]Embedding FunctionCLIP (for images), MiniLM (for text)Vector DBChromaDBOCR (if needed)pytesseract or easyocrFile HandlingPDF, TXT, JPG, PNGInterface (optional)Streamlit / FastAPI

### Project Structure

```text
multimodal-rag-janus/
├── ingest.py
├── query.py
├── main.py
├── data/
│   ├── documents/
│   │   ├── intro_ai.txt
│   │   └── chart.png
├── db/
├── utils/
│   ├── image_embedder.py
│   └── text_embedder.py
├── requirements.txt
```

### Step-by-Step Implementation

* **Step 1:** Install Dependencies

```bash
pip install chromadb sentence-transformers torchvision transformers pytesseract pillow
# If using deepseek-vl locally via Ollama:

ollama pull deepseek-vl
```

* **Step 2:** Embedding Helpers
* **Step 3:** Ingest Text & Image Files
* **Step 4:** Multimodal Query with DeepSeek-VL
* **Step 5:** Run It All

### Output
Ask questions like:

“What is described in the image file?”

“Summarize the main idea from the text and image combined.”

“What are the key topics covered in the chart?”

### Optional Extensions

* Use OCR to extract text from diagrams or scanned documents
* Add image captioning or vision-only Q&A
* Integrate Streamlit upload/chat UI
* Store metadata (file type, origin) for smarter tool use