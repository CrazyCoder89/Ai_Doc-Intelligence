# create_test_pdf.py
# This just creates a dummy PDF for testing purposes.
# You can delete this file after testing.

from reportlab.pdfgen import canvas

def create_test_pdf():
    # Create a PDF at this path
    path = "data/raw/test.pdf"
    
    # Create the PDF canvas
    c = canvas.Canvas(path)
    
    # Page 1
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "AI Document Intelligence System")
    c.setFont("Helvetica", 12)
    c.drawString(100, 700, "This is page 1 of our test document.")
    c.drawString(100, 680, "It contains information about Artificial Intelligence.")
    c.drawString(100, 660, "Machine Learning is a subset of AI.")
    c.drawString(100, 640, "Deep Learning uses neural networks with many layers.")
    
    # Page 2
    c.showPage()
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "Chapter 2: RAG Systems")
    c.setFont("Helvetica", 12)
    c.drawString(100, 700, "RAG stands for Retrieval Augmented Generation.")
    c.drawString(100, 680, "It combines search with language model generation.")
    c.drawString(100, 660, "First we retrieve relevant chunks from documents.")
    c.drawString(100, 640, "Then we pass them to an LLM to generate an answer.")
    
    # Save the PDF
    c.save()
    print("Test PDF created at data/raw/test.pdf")

if __name__ == "__main__":
    create_test_pdf()