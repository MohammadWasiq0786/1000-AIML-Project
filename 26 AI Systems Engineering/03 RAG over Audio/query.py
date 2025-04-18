import chromadb
import requests
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
 
def query_audio_knowledge(query: str, collection_name="audio_docs"):
    client = chromadb.Client()
    embedder = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.get_or_create_collection(name=collection_name, embedding_function=embedder)
    
    results = collection.query(query_texts=[query], n_results=3)
    context = "\n".join(results["documents"][0])
 
    prompt = f"Answer the following based on the audio transcript context:\n{context}\n\nQuestion: {query}"
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "mistral",
        "prompt": prompt,
        "stream": False
    })
    return response.json()["response"]