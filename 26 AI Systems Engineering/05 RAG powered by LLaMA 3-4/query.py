import chromadb
import requests
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
 
def query_llama_rag(user_query: str, collection_name="llama_rag_docs") -> str:
    client = chromadb.Client()
    embedder = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.get_or_create_collection(name=collection_name, embedding_function=embedder)
 
    results = collection.query(query_texts=[user_query], n_results=3)
    context = "\n".join(results["documents"][0])
 
    prompt = f"""Answer the question using only the context below:
 
Context:
{context}
 
Question:
{user_query}
 
Answer:"""
 
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    })
 
    return response.json()["response"]