from chat import chat_with_deepseek
from memory import reset_history
 
print("🤖 DeepSeek Mini-ChatGPT\nType 'exit' to quit or 'reset' to clear memory.")
 
while True:
    user_input = input("\nYou: ")
    if user_input.strip().lower() == "exit":
        break
    elif user_input.strip().lower() == "reset":
        reset_history()
        print("🔄 Memory cleared.")
        continue
 
    reply = chat_with_deepseek(user_input)
    print("\nAI:", reply)