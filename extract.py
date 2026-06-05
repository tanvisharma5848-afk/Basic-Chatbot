import fitz

docs = fitz.open('sample.pdf')
all_text = ""

for i in range(len(docs)):
    text = docs[i].get_text()
    all_text += text

with open('Extracted_Text.txt','w') as file:
    file.write(all_text)
    print("File Saved Successfully")