"""
Project 827. Document Processing Automation

Document processing automation extracts and structures information from documents like invoices, forms, and contracts. In this project, we’ll simulate extracting structured data (e.g., invoice number, date, total) from unstructured text using regular expressions.

This script automates the parsing of key fields from raw document text. For production, it can be enhanced with OCR (e.g., Tesseract), layout parsing, entity recognition (SpaCy), or document AI services like Amazon Textract or Google Document AI.
"""

import re
import pandas as pd
 
# Simulated raw text from scanned invoices (could come from OCR output)
documents = [
    "Invoice #12345\nDate: 2024-11-03\nTotal: $1,250.00\nCustomer: ABC Corp.",
    "Invoice #12346\nDate: 2024-11-04\nTotal: $2,100.50\nCustomer: XYZ Inc.",
    "Invoice #12347\nDate: 2024-11-05\nTotal: $785.75\nCustomer: GlobalTech Ltd."
]
 
# Function to extract structured fields from document text
def extract_invoice_data(text):
    invoice_no = re.search(r'Invoice\s*#(\d+)', text)
    date = re.search(r'Date:\s*([\d-]+)', text)
    total = re.search(r'Total:\s*\$([\d,]+\.\d{2})', text)
    customer = re.search(r'Customer:\s*(.+)', text)
 
    return {
        'InvoiceNumber': invoice_no.group(1) if invoice_no else None,
        'Date': date.group(1) if date else None,
        'TotalAmount': float(total.group(1).replace(',', '')) if total else None,
        'Customer': customer.group(1).strip() if customer else None
    }
 
# Extract data from each document
records = [extract_invoice_data(doc) for doc in documents]
 
# Create structured DataFrame
df = pd.DataFrame(records)
print("Extracted Invoice Data:")
print(df)