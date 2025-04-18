def get_reasoning_prompt(question: str) -> str:
    return f"""You are a logical reasoning AI. Solve the problem step by step using clear reasoning.
 
Problem: {question}
 
Let's think step by step:
"""