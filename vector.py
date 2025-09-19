from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import time
import json
import shutil
import pandas as pd

# Paths
db_location = "./chroma_roehampton_db"
meta_file_path = os.path.join(db_location, ".source_meta.json")

# CSV sources we rely on
csv_files = [
    "roehampton.csv",
    "undergraduate.csv",
    "postgraduate.csv",
]


def get_sources_meta() -> dict:
    """Return a dict containing mtimes for all CSV sources."""
    meta: dict[str, float] = {}
    for csv_path in csv_files:
        try:
            meta[csv_path] = os.path.getmtime(csv_path)
        except OSError:
            # If a file is missing, record zero so any later appearance triggers rebuild
            meta[csv_path] = 0.0
    return meta


def has_source_changed(existing_meta: dict | None, current_meta: dict) -> bool:
    if not existing_meta:
        return True
    for key, value in current_meta.items():
        if key not in existing_meta or float(existing_meta.get(key, 0)) != float(value):
            return True
    return False

def read_csv_with_fallback(path: str) -> pd.DataFrame:
    """Try multiple encodings to robustly read CSVs containing symbols like '£'."""
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    last_err: Exception | None = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as err:  # noqa: BLE001 - want to fall back through encodings
            last_err = err
            continue
    # If all fail, re-raise the last error for visibility
    raise last_err  # type: ignore[misc]


# Load CSVs (with encoding fallback)
roehampton_df = read_csv_with_fallback("roehampton.csv")
ug_df = read_csv_with_fallback("undergraduate.csv")
pg_df = read_csv_with_fallback("postgraduate.csv")

# Embeddings
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Determine if we need to (re)build the vector DB
current_meta = get_sources_meta()
stored_meta: dict | None = None
if os.path.exists(meta_file_path):
    try:
        with open(meta_file_path, "r", encoding="utf-8") as f:
            stored_meta = json.load(f)
    except Exception:
        stored_meta = None

# Rebuild if DB missing or sources changed
add_documents = not os.path.exists(db_location) or has_source_changed(stored_meta, current_meta)

if add_documents and os.path.exists(db_location):
    # Clean existing DB to avoid duplicate IDs and stale entries
    shutil.rmtree(db_location, ignore_errors=True)

if add_documents:
    documents = []
    ids = []

    # Roehampton general info
    for i, row in roehampton_df.iterrows():
        doc = Document(
            page_content=f"{row['Title']}: {row['Details']}",
            metadata={"type": "general"},
        )
        documents.append(doc)
        ids.append(f"general_{i}")

    # Undergraduate courses
    for i, row in ug_df.iterrows():
        course_name = row['Course']

        # Full info doc
        full_doc = Document(
            page_content=(
                f"Undergraduate Course: {course_name}\n"
                f"Next Entry: {row['Next Entry']}\n"
                f"Duration: {row['Duration']}\n"
                f"Start Date: {row['Start Date']}\n"
                f"Tuition Fee: {row['tuition fee']}"
            ),
            metadata={"type": "undergraduate", "course": course_name},
        )
        documents.append(full_doc)
        ids.append(f"ug_full_{i}")

        # Fee-only doc
        fee_doc = Document(
            page_content=f"Tuition fee for Undergraduate {course_name}: {row['tuition fee']}",
            metadata={"type": "undergraduate_fee", "course": course_name},
        )
        documents.append(fee_doc)
        ids.append(f"ug_fee_{i}")

    # Postgraduate courses
    for i, row in pg_df.iterrows():
        course_name = row['Course']

        # Full info doc
        full_doc = Document(
            page_content=(
                f"Postgraduate Course: {course_name}\n"
                f"Next Entry: {row['Next Entry']}\n"
                f"Duration: {row['Duration']}\n"
                f"Start Date: {row['Start Date']}\n"
                f"Tuition Fee: {row['Tuition fee']}"
            ),
            metadata={"type": "postgraduate", "course": course_name},
        )
        documents.append(full_doc)
        ids.append(f"pg_full_{i}")

        # Fee-only doc
        fee_doc = Document(
            page_content=f"Tuition fee for Postgraduate {course_name}: {row['Tuition fee']}",
            metadata={"type": "postgraduate_fee", "course": course_name},
        )
        documents.append(fee_doc)
        ids.append(f"pg_fee_{i}")

# Vector store
vector_store = Chroma(
    collection_name="roehampton_courses",
    persist_directory=db_location,
    embedding_function=embeddings,
)

# Insert on initial build or when sources changed
if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)
    # Persist meta so we can detect future changes
    os.makedirs(db_location, exist_ok=True)
    with open(meta_file_path, "w", encoding="utf-8") as f:
        json.dump(current_meta, f)

# Retriever (tuned for precise fee queries)
# Using MMR to diversify results and slightly higher k to capture fee-only docs
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 8,
        "fetch_k": 24,
    },
)
