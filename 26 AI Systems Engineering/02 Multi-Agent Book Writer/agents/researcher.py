import requests
from shared.context import context
 
def run_researcher():
    for chapter in context["chapters"]:
        prompt = f"Give background facts, quotes, and details for a chapter titled '{chapter}'."
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        })
        context["research"][chapter] = response.json()["response"]