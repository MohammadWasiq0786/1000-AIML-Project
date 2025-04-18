def get_rag_response(query: str) -> str:
    import requests
    response = requests.post("http://localhost:5000/ask", json={"query": query})
    return response.json()["response"]