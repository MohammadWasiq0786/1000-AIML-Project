from mcp import Agent, ToolRegistry
from tools.math_tool import MathTool
from tools.file_search_tool import FileSearchTool
 
tools = ToolRegistry()
tools.register(MathTool())
tools.register(FileSearchTool())
 
agent = Agent(
    name="local-agent",
    llm="http://localhost:11434/api/generate",  # Ollama local endpoint
    tools=tools,
    temperature=0.7,
)
 
if __name__ == "__main__":
    agent.serve(port=5000)