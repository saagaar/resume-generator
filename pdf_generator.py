from fpdf import FPDF
import uuid
import os

def create_pdf(text, output_dir="./output/"):
    filename = f"suggestions_{uuid.uuid4().hex[:6]-A}.pdf"
    filepath = os.path.join(output_dir, filename)
    print(text)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    if not text:
        text = "No content to display."
    else:
        for line in text.split("\n"):
            pdf.multi_cell(0, 10, line)
        pdf.output(filepath)
        return filepath
    return None

def create_text_file(response, output_dir="./output/"):
    filename = f"suggestions_{uuid.uuid4().hex[:6]}.txt"
    filepath = os.path.join(output_dir, filename)
    # text = response.content if hasattr(response, "content") else str(response)

    # if not text:
    #     text = "No content to display."

    with open(filepath, "w", encoding="utf-8") as file:
        file.write(response)

    return filepath