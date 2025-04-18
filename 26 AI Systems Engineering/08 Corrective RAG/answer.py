import requests
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
 
def get_rag_answer(query, collection_name="corrective_rag"):
    client = chromadb.Client()
    embedder = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.get_or_create_collection(name=collection_name, embedding_function=embedder)
 
    results = collection.query(query_texts=[query], n_results=3)
    context = "\n".join(results["documents"][0])
 
    prompt = f"""Use the following context to answer the question.
 
Context:
{context}
 
Question:
{query}
 
Answer:"""
 
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "mistral",
        "prompt": prompt,
        "stream": False
    })
 
    return response.json()["response"], context