# Project 2: Multi-Agent Book Writer

## Overview
This project simulates a writing team of AI agents that co-author a book. Each agent has a role:

* Planner: Defines chapters and structure
* Researcher: Finds or retrieves supporting content
* Writer: Drafts content for each chapter
* Editor: Reviews and polishes output

The agents communicate in sequence or loop using shared memory or direct calls, leading to a complete book draft.

## Tech Stack

### ComponentTool

LibLLMsOllama (mistral, llama3, deepseek)Agent OrchestrationPython classes or LangGraph (optional)Context SharingJSON file / in-memory objectTool Use (Optional)Web Search / Chroma for RAGOptional UIStreamlit or CLI

### Project Structure
```text
multiagent-book-writer/
├── main.py
├── agents/
│   ├── planner.py
│   ├── researcher.py
│   ├── writer.py
│   └── editor.py
├── shared/
│   └── context.py
├── output/
│   └── draft.txt
├── requirements.txt
└── config.yaml
```

### Step-by-Step Implementation

* **Step 1:** Install Dependencies
```bash
pip install openai ollama
ollama pull mistral  # or llama3, deepseek
```
We'll use Ollama’s local REST API (localhost:11434) to access the models.

* **Step 2:** Shared Context
* **Step 3:** Planner Agent
* **Step 4:** Researcher Agent
* **Step 5:** Writer Agent
* **Step 6:** Editor Agent
* **Step 7:** Run Pipeline

### Output

* A full multi-chapter book draft written and edited collaboratively by autonomous AI agents.
* Output is saved in output/draft.txt.

### Optional Extensions

* Use LangGraph for structured agent flow with retries
* Add memory and iteration feedback loop
* Add Streamlit UI to monitor each stage
* Export as PDF or Markdown