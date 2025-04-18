from ingest import ingest_multimodal
from query import query_multimodal_rag
 
print("Indexing multimodal documents...")
ingest_multimodal()
 
while True:
    question = input("\nAsk a question (or 'exit'): ")
    if question == "exit":
        break
    answer = query_multimodal_rag(question)
    print("\nAnswer:", answer)