from PyPDF2 import PdfReader
from docx import Document

def extract_text(path="cv/my_cv.pdf"):
    if path.endswith(".pdf"):
        with open(path, "rb") as file:
            reader = PdfReader(file)
            return "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
    elif path.endswith(".docx"):
        doc = Document(path)
        return "\n".join([p.text for p in doc.paragraphs])
    else:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()