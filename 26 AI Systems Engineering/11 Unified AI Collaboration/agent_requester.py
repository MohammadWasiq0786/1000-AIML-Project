import requests
import json
import uuid
 
# Load agent directory
with open('agent_directory.json') as f:
    directory = json.load(f)
 
tasks = [
    {"capability": "summarize", "text": "Google’s Agent2Agent protocol enables AI agents to collaborate effectively and share tasks securely within multi-agent environments."},
    {"capability": "generate_text", "text": "The future of AI agents"},
    {"capability": "sentiment_analysis", "text": "I'm thrilled with the results of using the new Agent2Agent protocol!"}
]
 
for task_info in tasks:
    capability = task_info["capability"]
    provider = next(agent for agent in directory['agents'] if capability in agent['capabilities'])
 
    payload = {
        "task_id": str(uuid.uuid4()),
        "text": task_info["text"]
    }
 
    endpoint = provider['endpoint']
    response = requests.post(endpoint, json=payload)
 
    if response.status_code == 200:
        result = response.json()
        print(f"\nProvider: {result['provider']}")
        print(f"Task ID: {result['task_id']}")
        print(f"Task Status: {result['status']}")
        print(f"Result: {result['result']}")
    else:
        print(f"Error contacting provider {provider['name']} (status: {response.status_code})")