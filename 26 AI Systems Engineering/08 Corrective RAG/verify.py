import requests
 
def verify_and_correct_answer(question, context, initial_answer):
    prompt = f"""You are an AI verifier. Review the following:
 
Question:
{question}
 
Context:
{context}
 
Initial Answer:
{initial_answer}
 
Does the answer match the context? If not, correct it. Be concise.
 
Verified Answer:"""
 
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "mistral",
        "prompt": prompt,
        "stream": False
    })
 
    return response.json()["response"]