import requests
from prompts.reasoning_templates import get_reasoning_prompt
 
def ask_question_with_reasoning(question):
    prompt = get_reasoning_prompt(question)
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "mistral",
        "prompt": prompt,
        "stream": False,
        "temperature": 0.4,
        "stop": ["\n\n"]
    })
    return response.json()["response"]
 
print("🤖 Reasoning AI Agent\nAsk logic/math/factual problems. Type 'exit' to quit.")
 
while True:
    user_input = input("\nProblem: ")
    if user_input.strip().lower() == "exit":
        break
    answer = ask_question_with_reasoning(user_input)
    print("\n🧠 Step-by-Step Answer:\n", answer)