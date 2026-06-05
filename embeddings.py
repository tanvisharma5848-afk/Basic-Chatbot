from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import pickle

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
)

with open('Chunks.pkl', 'rb') as file:
    texts = pickle.load(file)

vector_store = Chroma.from_texts(
    texts, embeddings,
    persist_directory="./chroma_langchain_db",
)

while True:
    query = input("User: ") #exit

    if query.lower() == "exit":
        print("Bye..")
        break

    results = vector_store.similarity_search(query, k =3)

    for i , index in enumerate(results , 1):
        print(f"{i}. {index.page_content}")