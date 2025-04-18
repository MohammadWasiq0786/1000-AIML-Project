conversation_history = []
 
def add_message(role, content):
    conversation_history.append({"role": role, "content": content})
 
def get_formatted_history():
    history = ""
    for msg in conversation_history:
        prefix = "You:" if msg["role"] == "user" else "AI:"
        history += f"{prefix} {msg['content']}\n"
    return history
 
def reset_history():
    conversation_history.clear()