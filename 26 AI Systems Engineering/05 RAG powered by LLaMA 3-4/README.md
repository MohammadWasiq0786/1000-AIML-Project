# Project 5: RAG powered by LLaMA 4

## Overview

You’ll build a document question-answering system where:

PDFs or text files are ingested and chunked

Each chunk is embedded and stored in ChromaDB

A user’s question is semantically matched to relevant chunks

The retrieved context is passed to LLaMA 4 (via Ollama) to generate an answer

## Tech Stack

### ComponentTool
/LibraryLLMOllama with llama3:latest (Meta's LLaMA 4 base)Vector DBChromaDBEmbeddingssentence-transformers (MiniLM)File IngestionPyMuPDF or plain .txt loaderUI (Optional)CLI / Streamlit / FastAPI

### Project Structure

```text
rag-llama4/
├── main.py
├── ingest.py
├── query.py
├── data/
│   └── sample_docs/
│       └── ai_intro.txt
├── db/
├── requirements.txt
```

### Step-by-Step Implementation

* **Step 1:** Install Dependencies

```bash
pip install chromadb sentence-transformers ollama
ollama pull llama3  # LLaMA 4 base
```

* **Step 2:** Ingest Documents into Vector DB
* **Step 3:** Query Engine using LLaMA 3/4
* **Step 4:** Run the Pipeline

### Output
You now have a RAG system that:

Accepts any .txt files as input

Lets users ask questions in natural language

Uses LLaMA 4 to answer based on retrieved, relevant document context

### Optional Extensions

* Switch to PDF ingestion using PyMuPDF
* Add Streamlit chat UI with document upload
* Expand to multi-modal support (prep for Project 6!)
* Save session history with SQLite or JSON