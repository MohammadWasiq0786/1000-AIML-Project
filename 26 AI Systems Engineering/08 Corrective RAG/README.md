# Project 8: Corrective RAG

## Overview
This project improves a RAG pipeline by adding a second-pass verifier agent that:

Evaluates the initial answer

Checks for factual consistency with retrieved context

Corrects or confirms the final output

You’ll simulate how multi-step reasoning and self-reflection can improve answer quality.

## Tech Stack

### ComponentTool
LibraryLLMOllama with mistral / llama3 / deepseekVector DBChromaDBEmbeddingssentence-transformers (MiniLM)Feedback AgentSame LLM, different promptUI (Optional)CLI or Streamlit

### Project Structure

```text
corrective-rag/
├── main.py
├── ingest.py
├── answer.py
├── verify.py
├── data/
│   └── context_docs/
├── db/
├── requirements.txt
```

### Step-by-Step Implementation

* **Step 1:** Install Dependencies

```bash
pip install chromadb sentence-transformers ollama
ollama pull mistral  # or llama3 / deepseek
```

* **Step 2:** Ingest Contextual Documents
* **Step 3:** Initial RAG Answer
* **Step 4:** Verifier Pass
* **Step 5:** Run Corrective RAG Pipeline

### Output
You now have a Corrective RAG agent that:

Answers based on local document context

Automatically critiques and corrects its own response

Uses no API keys, all local tools

### Optional Extensions

Add a "confidence rating" in the verifier's output

Save versions: user question, initial answer, corrected answer

Add streamed response from LLM

Use multiple verifier agents (ensemble checking)