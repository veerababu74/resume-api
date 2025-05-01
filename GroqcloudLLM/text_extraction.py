import os
import re
from PyPDF2 import PdfReader
from docx import Document


def extract_text_from_txt(file_path):
    """Extract text from a .txt file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def extract_text_from_pdf(file_path):
    """Extract text from a .pdf file."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def extract_text_from_docx(file_path):
    """Extract text from a .docx file."""
    doc = Document(file_path)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    return text


def clean_text(text):
    """
    Clean the extracted text by removing extra spaces, lines, and unnecessary \n.
    """
    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)
    # Remove leading/trailing spaces
    text = text.strip()
    return text


def extract_and_clean_text(file_path):
    """Extract and clean text based on the file type."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    _, extension = os.path.splitext(file_path)
    extension = extension.lower()

    if extension == ".txt":
        raw_text = extract_text_from_txt(file_path)
    elif extension == ".pdf":
        raw_text = extract_text_from_pdf(file_path)
    elif extension == ".docx":
        raw_text = extract_text_from_docx(file_path)
    else:
        raise ValueError(f"Unsupported file format: {extension}")

    # Clean the extracted text
    cleaned_text = clean_text(raw_text)
    return cleaned_text


# Example usage
# if __name__ == "__main__":
#     file_path = "t.docx"
#     try:
#         cleaned_text = extract_and_clean_text(file_path)
#         print("\nCleaned Extracted Text:")
#         print(cleaned_text)
#     except Exception as e:
#         print(f"Error: {e}")
