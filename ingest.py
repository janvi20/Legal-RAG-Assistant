import os
import json
from typing import List
from dotenv import load_dotenv
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


load_dotenv()


# -----------------------------
# Configuration
# -----------------------------

DATASET_NAME = "common-pile/caselaw_access_project"

RAW_DATA_PATH = "docs/cases.json"

VECTOR_DB_PATH = "db/chroma_db"

COLLECTION_NAME = "caselaw_rag"


_embedding_model = None


# -----------------------------
# Embedding Model
# -----------------------------

def get_embedding_model():

    """
    Load free HuggingFace embedding model once.
    """

    global _embedding_model

    if _embedding_model is None:

        print("Loading embedding model...")

        _embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",

            model_kwargs={"device": "cpu"},

            encode_kwargs={"normalize_embeddings": True}
        )

    return _embedding_model


# -----------------------------
# Fetch Dataset
# -----------------------------

def fetch_cases(limit: int = 50) -> List[dict]:

    print(f"Fetching {limit} cases...")

    dataset = load_dataset(DATASET_NAME, streaming=True, split="train")

    cases = list(dataset.take(limit))

    if not cases:
        raise ValueError("No cases fetched")

    print(f"Fetched {len(cases)} cases")

    return cases


# -----------------------------
# Save Raw Dataset
# -----------------------------

def save_cases(cases: List[dict], output_path: str = RAW_DATA_PATH):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)


    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)

    print(f"Saved raw cases -> {output_path}")


# -----------------------------
# Convert to LangChain Documents
# -----------------------------

def create_documents(cases: List[dict]) -> List[Document]:

    documents = []

    for case in cases:
        text = case.get("text", "")

        if not text.strip():
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "case_id": case.get(
                        "id"
                    ),

                    "title": case.get(
                        "title"
                    ),

                    "source": DATASET_NAME
                }
            )
        )


    print(f"Created {len(documents)} documents")

    return documents


# -----------------------------
# Chunk Documents
# -----------------------------

def split_documents(documents: List[Document]):

    print("Splitting documents...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    return chunks


# -----------------------------
# Create Chroma Vector Store
# -----------------------------

def create_vector_store(chunks: List[Document]):

    print("Creating Chroma vector store...")

    embeddings = get_embedding_model()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH,
        collection_name=COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print(f"Vector DB saved -> {VECTOR_DB_PATH}")

    return vectorstore


# -----------------------------
# Complete Ingestion Pipeline
# -----------------------------

def run_ingestion(limit=50):

    print("\n=== Legal RAG Ingestion Started ===\n")

    # 1. Fetch cases
    cases = fetch_cases(limit)

    # 2. Save backup JSON
    save_cases(cases)

    # 3. Convert to documents
    documents = create_documents(cases)

    # 4. Chunk documents
    chunks = split_documents(documents)

    # 5. Create embeddings + vector DB
    vectorstore = create_vector_store(chunks)

    print("\n=== Ingestion Complete ===")

    return vectorstore


# -----------------------------
# Run
# -----------------------------

if __name__ == "__main__":

    vector_db = run_ingestion(limit=50)