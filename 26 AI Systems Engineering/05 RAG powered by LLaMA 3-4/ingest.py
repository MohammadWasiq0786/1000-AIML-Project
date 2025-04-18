import os
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
 
def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
 
def ingest_documents(folder="data/sample_docs", collection_name="llama_rag_docs"):
    client = chromadb.Client()
    embedder = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.get_or_create_collection(name=collection_name, embedding_function=embedder)
 
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                text = f.read()
                chunks = chunk_text(text)
                ids = [f"{file}_chunk{i}" for i in range(len(chunks))]
                collection.add(documents=chunks, ids=ids)