from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
 
model_id = "deepseek-ai/deepseek-llm-7b-base"
 
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
 
dataset = load_dataset("json", data_files="dataset/finetune_data.jsonl")["train"]
 
def format(sample):
    prompt = f"### Instruction:\n{sample['instruction']}\n\n### Response:\n{sample['output']}"
    return {"input_ids": tokenizer(prompt, truncation=True, padding="max_length", max_length=512)["input_ids"]}
 
dataset = dataset.map(format)
 
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.float16
)
 
model = prepare_model_for_kbit_training(model)
 
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
 
model = get_peft_model(model, lora_config)
 
training_args = TrainingArguments(
    output_dir="./adapters",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_dir="./logs",
    save_total_limit=2,
    fp16=True,
    report_to="none"
)
 
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)
 
trainer.train()
model.save_pretrained("adapters")