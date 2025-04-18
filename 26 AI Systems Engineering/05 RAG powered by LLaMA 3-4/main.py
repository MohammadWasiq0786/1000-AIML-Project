from ingest import ingest_documents
from query import query_llama_rag
 
# Step 1: Index documents
print("Indexing documents...")
ingest_documents()
 
# Step 2: Query
while True:
    user_input = input("\nAsk something about the documents (or type 'exit'): ")
    if user_input.lower() == "exit":
        break
    answer = query_llama_rag(user_input)
    print("\nAnswer:", answer)