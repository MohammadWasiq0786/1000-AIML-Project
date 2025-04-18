from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
 
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-llm-7b-base", device_map="auto", load_in_4bit=True)
model = PeftModel.from_pretrained(model, "adapters")
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-base")
 
prompt = "### Instruction:\nWho is the author of 1984?\n\n### Response:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
 
output = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))