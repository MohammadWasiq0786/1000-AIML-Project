from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
 
def get_text_embedder():
    return SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")