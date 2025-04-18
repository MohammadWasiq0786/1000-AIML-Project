import requests
from memory import get_formatted_history, add_message
 
def chat_with_deepseek(user_input):
    add_message("user", user_input)
 
    prompt = get_formatted_history() + "\nAI:"
    
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "deepseek",
        "prompt": prompt,
        "stream": False,
        "temperature": 0.7,
    })
 
    answer = response.json()["response"]
    add_message("assistant", answer)
    return answer