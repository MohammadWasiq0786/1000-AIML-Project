from mcp.tools import Tool
 
class FileSearchTool(Tool):
    def __init__(self):
        super().__init__(name="file_search", description="Search inside a file.")
        with open("sample.txt", "r") as f:
            self.content = f.read()
 
    def invoke(self, input_text: str) -> str:
        if input_text.lower() in self.content.lower():
            return f"Found '{input_text}' in document."
        return f"'{input_text}' not found in document."