from mcp import Agent, ToolRegistry
from tools.chromadb_tool import ChromaTool
from rag_agent import get_rag_response
 
# Register your tool (vector search tool for RAG)
tools = ToolRegistry()
tools.register(ChromaTool())
 
agent = Agent(
    name="rag-agent",
    llm="http://localhost:11434/api/generate",  # Ollama endpoint
    tools=tools,
    max_tokens=1024,
    temperature=0.7,
)
 
if __name__ == "__main__":
    agent.serve(port=5000)