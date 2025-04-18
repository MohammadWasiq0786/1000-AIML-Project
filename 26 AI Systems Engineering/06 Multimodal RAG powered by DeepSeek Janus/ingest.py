import os
import chromadb
from utils.text_embedder import get_text_embedder
from utils.image_embedder import CLIPImageEmbedder
 
def ingest_multimodal(folder="data/documents", collection_name="janus_docs"):
    client = chromadb.Client()
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=get_text_embedder()
    )
 
    clip_embedder = CLIPImageEmbedder()
    i = 0
 
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        if fname.endswith(".txt"):
            with open(fpath, "r", encoding="utf-8") as f:
                text = f.read()
                collection.add(documents=[text], ids=[f"text-{i}"])
                i += 1
        elif fname.endswith((".jpg", ".png")):
            embedding = clip_embedder.get_embedding(fpath)
            collection.add(documents=[f"Image file: {fname}"], ids=[f"image-{i}"], embeddings=[embedding])
            i += 1