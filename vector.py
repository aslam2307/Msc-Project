from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import pandas as pd
import os

# Initialize embedding model
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Vector DB path
db_location = "./chroma_roehampton_db"
add_documents = not os.path.exists(db_location)

documents = []
ids = []

if add_documents:
    # ---------- General Info ----------
    df_general = pd.read_csv("roehampton.csv", nrows=7)
    for i, row in df_general.iterrows():
        text = f"{row['Section']}: {row['Detail']}"
        documents.append(Document(page_content=text, metadata={"section": row['Section']}, id=f"general_{i}"))
        ids.append(f"general_{i}")

    # ---------- Undergraduate Courses ----------
    df_undergrad = pd.read_csv("roehampton.csv", skiprows=8, nrows=22)
    for i, row in df_undergrad.iterrows():
        course = row["Undergraduate Courses"]
        details = "; ".join([f"{col}: {row[col]}" for col in df_undergrad.columns if col != "Undergraduate Courses"])
        text = f"Undergraduate Course - {course}: {details}"
        documents.append(Document(page_content=text, metadata={"type": "undergraduate"}, id=f"ug_{i}"))
        ids.append(f"ug_{i}")

    # ---------- Postgraduate Courses ----------
    df_postgrad = pd.read_csv("roehampton.csv", skiprows=31)
    for i, row in df_postgrad.iterrows():
        course = row["Postgraduate Courses"]
        details = "; ".join([f"{col}: {row[col]}" for col in df_postgrad.columns if col != "Postgraduate Courses"])
        text = f"Postgraduate Course - {course}: {details}"
        documents.append(Document(page_content=text, metadata={"type": "postgraduate"}, id=f"pg_{i}"))
        ids.append(f"pg_{i}")

# Create or load Chroma vector DB
vector_store = Chroma(
    collection_name="roehampton_qa",
    persist_directory=db_location,
    embedding_function=embeddings,
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})
