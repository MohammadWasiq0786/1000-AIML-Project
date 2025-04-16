"""
Project 461. Medical chatbot implementation
Description:
A medical chatbot assists users with health-related queries, symptoms, and advice. It can provide general information, recommend next steps, or route users to professionals. In this project, we'll implement a rule-based + LLM hybrid chatbot that simulates a basic symptom checker with natural responses.

✅ What It Does:
Uses rule-based logic for known symptoms.

Falls back to GPT-2 for conversational replies.

Can be extended to:

Use BioGPT, MedAlpaca, or ChatGPT API

Integrate with FHIR data, triage tools, or appointment scheduling

Add speech input/output for voice-driven bots

For real-world systems:

Use LangChain, Rasa, or Haystack

Integrate clinical knowledge bases like SNOMED CT, UMLS, or MedlinePlus
"""


from transformers import pipeline
 
# 1. Load a question-answering pipeline (general-purpose LLM)
qa = pipeline("text-generation", model="gpt2")
 
# 2. Rule-based triage logic (simplified)
def rule_based_response(user_input):
    if "fever" in user_input.lower() and "cough" in user_input.lower():
        return "It could be a respiratory infection. Monitor symptoms and consider seeing a doctor if it worsens."
    elif "headache" in user_input.lower() and "nausea" in user_input.lower():
        return "These may be signs of a migraine. Try rest and hydration, and consult a physician if persistent."
    else:
        return None
 
# 3. Chatbot interaction loop
print("Welcome to MedBot! Ask your symptom-related questions. Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    
    # First try rule-based logic
    response = rule_based_response(user_input)
    
    # If no rule match, fallback to LLM-generated response
    if response is None:
        response = qa(user_input, max_length=50, num_return_sequences=1)[0]["generated_text"]
    
    print("MedBot:", response)