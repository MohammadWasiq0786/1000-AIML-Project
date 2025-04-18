import requests
from shared.context import context
 
def run_writer():
    for chapter in context["chapters"]:
        prompt = f"Using the following research:\n{context['research'][chapter]}\nWrite the full draft of the chapter titled '{chapter}'."
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        })
        context["drafts"].append(f"# {chapter}\n{response.json()['response']}\n\n")