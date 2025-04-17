"""
Project 537: Paraphrase Generation
Description:
Paraphrase generation involves creating a new sentence that has the same meaning as the original one but with different wording. This task is useful for creating variations of content while maintaining the original intent. In this project, we will use a transformer model for generating paraphrases of a given sentence.
"""

from transformers import pipeline
 
# 1. Load pre-trained model for paraphrase generation
paraphraser = pipeline("text2text-generation", model="t5-small")
 
# 2. Provide a sentence to generate a paraphrase
sentence = "The quick brown fox jumps over the lazy dog."
 
# 3. Use the model to generate a paraphrase
paraphrased_text = paraphraser(f"paraphrase: {sentence}", max_length=100, num_return_sequences=1)
 
# 4. Display the paraphrased text
print(f"Paraphrased Text: {paraphrased_text[0]['generated_text']}")