import fitz
import os

def read_files_from_folder(folder_path):
    file_data = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(folder_path, file_name)
            try:
                content = extract_text_from_pdf(file_path)
                file_data.append({"file_name": file_name, "content": content})
            except Exception as e:
                print(f"Error processing file: {file_name}")
                print(f"Error message: {str(e)}\n")

    return file_data

def extract_text_from_pdf(pdf_path):
    text = ''
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text