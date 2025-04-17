"""
Project 539: Table-to-Text Generation
Description:
Table-to-text generation involves converting structured tabular data into natural language text. This is useful for automatically generating summaries from data such as financial reports, sports scores, and sales figures. In this project, we will use a transformer model to generate text based on tabular data.
"""

from transformers import pipeline
 
# 1. Load pre-trained model for table-to-text generation
table_to_text_generator = pipeline("text2text-generation", model="t5-small")
 
# 2. Define tabular data (e.g., sales data for a month)
sales_data = {
    'product': ['Product A', 'Product B', 'Product C'],
    'sales': [5000, 7000, 3000],
    'profit': [1500, 2000, 1000]
}
 
# 3. Convert the tabular data into a text prompt
table_input = f"Sales for the month: {sales_data['product'][0]} sold {sales_data['sales'][0]} units with a profit of {sales_data['profit'][0]}, {sales_data['product'][1]} sold {sales_data['sales'][1]} units with a profit of {sales_data['profit'][1]}, and {sales_data['product'][2]} sold {sales_data['sales'][2]} units with a profit of {sales_data['profit'][2]}."
 
# 4. Use the model to generate text based on the tabular data
generated_text = table_to_text_generator(table_input, max_length=100, num_return_sequences=1)
 
# 5. Display the generated text
print(f"Generated Text: {generated_text[0]['generated_text']}")