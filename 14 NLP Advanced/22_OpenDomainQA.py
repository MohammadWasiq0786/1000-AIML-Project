"""
Project 542: Open Domain Question Answering
Description:
Open domain question answering (QA) systems answer questions posed by users without being restricted to a specific domain. These systems need to understand natural language and retrieve relevant information from large amounts of unstructured text. In this project, we will use a pre-trained transformer model like T5 or BERT to build a simple open domain question answering system.
"""

from transformers import pipeline
 
# 1. Load the pre-trained question answering model
qa_pipeline = pipeline("question-answering")
 
# 2. Provide a context and a question
context = """
Open domain question answering systems are designed to answer any question, regardless of the subject matter. 
They achieve this by retrieving relevant information from a large corpus of unstructured data. These systems use 
techniques such as natural language processing (NLP) and machine learning to provide accurate and relevant answers.
"""
question = "What are open domain question answering systems?"
 
# 3. Use the pipeline to get the answer from the context
result = qa_pipeline(question=question, context=context)
 
# 4. Display the result
print(f"Answer: {result['answer']}")