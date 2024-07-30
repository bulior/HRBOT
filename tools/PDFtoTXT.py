import os
from PyPDF2 import PdfReader

def pdf_to_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

def convert_pdfs_in_folder(folder_path):
    txt_folder_path = os.path.join(folder_path, '../txt')
    
    if not os.path.exists(txt_folder_path):
        os.makedirs(txt_folder_path)
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            text = pdf_to_text(pdf_path)
            txt_filename = filename.replace('.pdf', '.txt')
            txt_path = os.path.join(txt_folder_path, txt_filename)
            with open(txt_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(text)
            print(f"Converted {filename} to {txt_filename}")

docsfolder_path = 'docs/uploaded_documents'  # Ersetze dies durch den tatsächlichen Pfad zum Ordner
#txt_folder_path = os.path.join(docsfolder_path, 'txt')
convert_pdfs_in_folder(docsfolder_path)