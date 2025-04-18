import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
 
def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
 
def ingest_text_to_chroma(text, collection_name="audio_docs"):
    client = chromadb.Client()
    embedder = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.get_or_create_collection(name=collection_name, embedding_function=embedder)
 
    chunks = chunk_text(text)
    ids = [f"chunk-{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids)