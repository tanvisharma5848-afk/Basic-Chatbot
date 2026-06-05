from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

#Load GROQ model.
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
    max_retries=2,
)

#Embedding Model.
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
)

#VectorStore.
vector_store = Chroma(persist_directory="./chroma_langchain_db",embedding_function=embeddings)


while True:
    query = input("User: ")
    if query.lower() == "exit":
        break
    
    results = vector_store.similarity_search(query, k=3)
    context = "\n".join([i.page_content for i in results])

    prompt = f"""
Answer this question based on the context below.
Context:
{context}

Question: {query}

Answer:
"""

    response = model.invoke(prompt)
    print("AI:", response.content)