from mcp.tools import Tool
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
 
class ChromaTool(Tool):
    def __init__(self):
        super().__init__(name="document_search", description="Search over documents")
        self.client = chromadb.Client()
        self.embedding_func = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.collection = self.client.create_collection(name="docs", embedding_function=self.embedding_func)
 
    def invoke(self, input_text: str) -> str:
        results = self.collection.query(query_texts=[input_text], n_results=3)
        return "\n\n".join([doc for doc in results["documents"][0]])


# Step 4:

from chromadb import Client
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
 
embedding_func = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = Client()
collection = client.create_collection(name="docs", embedding_function=embedding_func)
 
# Load sample text or PDF
docs = ["Artificial Intelligence is revolutionizing many industries...", "MCP enables modular tool use for AI agents..."]
collection.add(documents=docs, ids=["doc1", "doc2"])