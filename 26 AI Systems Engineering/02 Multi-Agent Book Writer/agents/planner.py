import requests
from shared.context import context
 
def run_planner():
    prompt = f"Plan 5 chapters for a book titled '{context['title']}'. Return a list."
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "mistral",
        "prompt": prompt,
        "stream": False
    })
    context["chapters"] = response.json()["response"].strip().split("\n")