"""
Project 459. Medical report generation
Description:
Medical report generation involves automatically creating diagnostic or radiology reports from medical images or clinical notes. In this project, we'll simulate a text generation system that takes in a brief patient case description and generates a summary report using a Transformer-based language model.

About:
✅ What It Does:
Accepts a patient summary and generates a radiology-style report continuation.

Uses DistilGPT2 here, but can be replaced with domain-specific models like BioGPT, MedPalm, or ClinicalT5.

Can be extended to:

Generate reports directly from image embeddings

Train on real radiology report corpora

Add sectioned outputs (e.g., Impression, Findings)

For real use cases, you can fine-tune models like:

BioGPT, PubMedBERT, or T5 on MIMIC-CXR reports

Use datasets like IU X-ray, MIMIC-CXR, OpenI
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
 
# 1. Load a pretrained language model (GPT2 for simulation)
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
 
# 2. Simulated clinical case input (could be output from a radiology image model)
clinical_input = (
    "Patient is a 68-year-old male with a history of smoking presenting with shortness of breath. "
    "Chest X-ray shows right upper lobe opacity and mild pleural effusion. Findings are suspicious for..."
)
 
# 3. Encode and generate report continuation
inputs = tokenizer.encode(clinical_input, return_tensors="pt")
outputs = model.generate(
    inputs,
    max_length=150,
    num_return_sequences=1,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.8
)
 
# 4. Decode and display report
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated Medical Report:\n")
print(generated_text)