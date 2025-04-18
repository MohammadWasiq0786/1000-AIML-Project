# Project 7: A mini-ChatGPT using DeepSeek-R1

## Overview
You’ll build a simple chatbot with:

Conversational memory

Local inference using DeepSeek-R1

Optional frontend (CLI or Streamlit)

The core focus: maintain back-and-forth context and simulate an intelligent assistant that runs 100% locally.

## Tech Stack

### ComponentTool
LibraryLLMdeepseek-r1 via OllamaBackendPython + REST clientMemorySimple list-based context (JSON/SQLite optional)UI (Optional)Streamlit / CLI

### Project Structure

```text
mini-chatgpt-deepseek/
├── main.py
├── chat.py
├── memory.py
├── requirements.txt
```

### Step-by-Step Implementation

* **Step 1:** Install Dependencies & Model

```bash
pip install ollama
ollama pull deepseek:base  # for DeepSeek-R1
# (If DeepSeek-R1 is not yet available under that name, use: deepseek or deepseek-coder.)
```

* **Step 2:** Create Conversation Memory Handler
* **Step 3:** Chat Logic with DeepSeek
* **Step 4:** CLI Runner

### Output
Runs like a basic ChatGPT, locally.

Maintains chat context with each new turn.

No API keys or cloud dependencies.

### Optional Enhancements

Add Streamlit UI with chat history

Save conversations in JSON or SQLite

Add tools (web search, calc, etc.) via MCP

Enable voice input/output (whisper + TTS)