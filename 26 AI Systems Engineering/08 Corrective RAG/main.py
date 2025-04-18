from ingest import ingest_docs
from answer import get_rag_answer
from verify import verify_and_correct_answer
 
# Step 1: Load data into Chroma
print("Indexing documents...")
ingest_docs()
 
# Step 2: Ask a question
while True:
    user_question = input("\nAsk a question (or type 'exit'): ")
    if user_question.lower() == "exit":
        break
 
    print("\n🔍 Generating initial answer...")
    initial_answer, context = get_rag_answer(user_question)
    print("\n🧠 Initial Answer:\n", initial_answer)
 
    print("\n🔎 Verifying and correcting...")
    verified = verify_and_correct_answer(user_question, context, initial_answer)
    print("\n✅ Final Answer:\n", verified)