from mcp.tools import Tool
 
class MathTool(Tool):
    def __init__(self):
        super().__init__(
            name="math_tool",
            description="Performs basic arithmetic operations. Usage: add, subtract, multiply, divide."
        )
 
    def invoke(self, input_text: str) -> str:
        try:
            result = eval(input_text, {"__builtins__": {}})
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {e}"