import requests
import chromadb
from utils.text_embedder import get_text_embedder
 
def query_multimodal_rag(query, collection_name="janus_docs"):
    client = chromadb.Client()
    collection = client.get_or_create_collection(name=collection_name, embedding_function=get_text_embedder())
 
    results = collection.query(query_texts=[query], n_results=3)
    context = "\n".join(results["documents"][0])
 
    prompt = f"""You are a multimodal AI assistant. Answer the question using the following context:
    
{context}
 
Question: {query}
Answer:"""
 
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "deepseek-vl",
        "prompt": prompt,
        "stream": False
    })
    
    return response.json()["response"]