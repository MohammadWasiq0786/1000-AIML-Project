# Project 1: MCP-powered Agentic RAG

## Overview
In this project, you'll build a local agentic RAG system using the Model Context Protocol (MCP) to modularly connect an LLM to external tools like web search, vector DBs, or file loaders. The system retrieves relevant context, injects it into prompts, and lets the agent reason autonomously.

## Tech Stack

### ComponentTool

LibLanguage ModelOllama (e.g., mistral, llama3)Agent Frameworkmcp Python library (local server)RAG PipelineLangChain or raw orchestrationVector StoreChromaDB (local, open-source)File HandlingPDF/Text LoaderFrontend (optional)Streamlit or CLIEnvironmentPython 3.10+, virtualenv or Conda

### Project Structure

```text
agentic-rag-mcp/
├── main.py
├── mcp_config.yaml
├── vector_store/
├── data/
│   └── sample_docs/
├── tools/
│   └── chromadb_tool.py
├── rag_agent.py
└── requirements.txt
```

### Step-by-Step Implementation

* **Step 1:** Set up your local environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install mcp ollama langchain chromadb pypdf
```

* Start the Ollama server and pull a model:
```bash
ollama serve
ollama pull mistral  # or llama3
```

* **Step 2:** Launch a local MCP Server
* **Step 3:** Define a Vector Search Tool
* **Step 4:** Load Docs into Chroma Vector DB
* **Step 5:** Agent Query Logic (Optional)


Then from another terminal or via a frontend:
```py
from rag_agent import get_rag_response
print(get_rag_response("What is the purpose of the MCP protocol?"))
```

### Optional Extensions

* Add Streamlit frontend
* Add PDF/text drag & drop upload for ingestion
* Integrate Web Search Tool for fallback queries
* Add agent memory (JSON store)