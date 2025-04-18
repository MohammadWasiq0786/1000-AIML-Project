# Project 4: Build a Local MCP Server

## Overview

This project sets up a local MCP server with one or more tools. You’ll configure:

A local LLM (via Ollama)

An MCP-compatible agent that can invoke tools

Custom tools like document search or calculator

REST API to send queries to the agent

## Tech Stack

### ComponentToolAgent
Runtimemcp Python SDKLLM BackendOllama with mistral or llama3ToolsCustom Python tools (e.g. search, RAG, math)InterfacePython or HTTP POST

### Project Structure

```text
local-mcp-server/
├── main.py
├── tools/
│   ├── math_tool.py
│   └── file_search_tool.py
├── requirements.txt
└── README.md
```

### Step-by-Step Implementation

* **Step 1:** Install Dependencies

```bash
pip install mcp ollama
ollama pull mistral  # or llama3, deepseek, etc.
```

* **Step 2: Build a Tool (e.g., Math Calculator)

* **Step 3:** (Optional) Add Another Tool – File Search
* **Step 4:** Start the MCP Server

### Output
* Your local MCP agent now responds to queries and invokes tools based on prompt needs.
* Try:
    * "Use the math_tool to calculate 75/3"
    * "Use the file_search tool to check if the word 'robot' is in the file"

### Optional Add-ons

* Add tools like: Wolfram, Wikipedia, LangChain-style RAG, weather checker
* Wrap it with FastAPI or Streamlit for local UI
* Enable logging + memory with a JSON file