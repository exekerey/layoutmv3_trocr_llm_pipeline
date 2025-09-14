import pytesseract
from pdf2image import convert_from_path

pdf_path = "data/199.pdf"
pages = convert_from_path(pdf_path, dpi=300)

text = ""
for page in pages:
    text += pytesseract.image_to_string(page, lang="rus+eng") + "\n"
    print(pytesseract.image_to_boxes(page, lang="rus+eng"))

print(text)
