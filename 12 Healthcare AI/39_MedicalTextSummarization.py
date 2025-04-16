"""
Project 479. Medical text summarization
Description:
Medical text summarization automatically condenses lengthy clinical notes, discharge summaries, or radiology reports into concise, informative summaries. This aids doctors in quickly understanding patient history or findings. In this project, we use a transformer-based model (e.g., T5) to perform abstractive summarization of medical notes.

About:
✅ What It Does:
Uses T5 model to generate an abstractive summary of a long clinical note.

Outputs a brief, human-readable summary that preserves the key details.

Can be extended to:

Use BioBART, ClinicalT5, or LongT5 for better performance

Fine-tune on datasets like MIMIC-CXR or i2b2

Integrate in EHR systems to show auto-summaries for physicians

You can fine-tune models using:

MIMIC-III Notes, PubMed abstracts, or clinical summaries
"""

from transformers import T5Tokenizer, T5ForConditionalGeneration
 
# 1. Load T5 model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")
 
# 2. Simulated clinical note (can be a discharge summary or radiology report)
clinical_note = """
Patient is a 72-year-old female admitted for worsening dyspnea and fatigue.
Chest X-ray revealed bilateral pulmonary infiltrates.
Started on IV antibiotics and oxygen therapy. 
Comorbidities include type 2 diabetes, hypertension, and chronic kidney disease stage 3.
"""
 
# 3. Prepare input with "summarize:" prefix
input_text = "summarize: " + clinical_note
inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
 
# 4. Generate summary
summary_ids = model.generate(inputs, max_length=50, num_beams=4, early_stopping=True)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
 
# 5. Display summary
print("Clinical Summary:\n")
print(summary)