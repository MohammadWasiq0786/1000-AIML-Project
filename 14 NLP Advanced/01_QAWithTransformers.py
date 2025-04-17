"""
Project 521: Question Answering with Transformers
Description:
Question answering (QA) is a task in NLP where a system answers questions based on a given text. In this project, we will implement a question answering system using transformer models (e.g., BERT or DistilBERT) to extract answers from context paragraphs.
"""

from transformers import pipeline
 
# 1. Load the pre-trained question answering model
qa_pipeline = pipeline("question-answering")
 
# 2. Provide the context and the question
context = """
Transformers are a deep learning model architecture introduced in the paper "Attention is All You Need" in 2017. 
They have been widely used in NLP tasks such as machine translation, text summarization, and question answering. 
Transformers rely on attention mechanisms to process input data in parallel, as opposed to sequential models like RNNs and LSTMs.
"""
question = "What are transformers used for?"
 
# 3. Use the pipeline to get the answer
result = qa_pipeline(question=question, context=context)
 
# 4. Display the result
print(f"Answer: {result['answer']}")