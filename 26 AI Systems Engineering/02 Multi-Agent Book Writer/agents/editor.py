import requests
from shared.context import context
 
def run_editor():
    final_output = ""
    for draft in context["drafts"]:
        prompt = f"Edit the following for grammar and coherence:\n{draft}"
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        })
        final_output += response.json()["response"] + "\n\n"
    
    with open("output/draft.txt", "w") as f:
        f.write(final_output)