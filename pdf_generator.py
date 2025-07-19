from fpdf import FPDF
import uuid
import os

def create_pdf(text, output_dir="."):
    filename = f"suggestions_{uuid.uuid4().hex[:6]}.pdf"
    filepath = f"{output_dir}/{filename}"

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for line in text.split("\n"):
        pdf.multi_cell(0, 10, line)

    pdf.output(filepath)
    return filepath