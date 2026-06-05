from langchain_text_splitters import RecursiveCharacterTextSplitter
import pickle

with open('Extracted_Text.txt','r') as file:
    text = file.read()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_text(text)

with open('Chunks.pkl','wb') as file:
    pickle.dump(texts,file)
    print("File Saved Successfully")