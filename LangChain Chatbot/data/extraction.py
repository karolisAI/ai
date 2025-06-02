import os
from PyPDF2 import PdfReader

def extract_text_from_pdfs(directory):
    pdf_texts = []
    for filename in os.listdir(directory):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'rb') as file:
                reader = PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                pdf_texts.append({
                    'file_name': filename,
                    'content': text
                })
    return pdf_texts