from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from langchain_core.documents import Document
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

from langchain_community.retrievers import BM25Retriever

from sentence_transformers import CrossEncoder


# ==================================================
# Configuration
# ==================================================

VECTOR_DB_PATH = "db/chroma_db"

COLLECTION_NAME = "caselaw_rag"

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

RERANK_MODEL = "BAAI/bge-reranker-base"



# ==================================================
# Embedding Model
# ==================================================

print("Loading embedding model...")

embedding_model = HuggingFaceEmbeddings(

    model_name=EMBEDDING_MODEL,

    model_kwargs={
        "device": "cpu"
    },

    encode_kwargs={
        "normalize_embeddings": True
    }
)



# ==================================================
# Chroma Vector Database
# ==================================================

print("Loading ChromaDB...")


vector_db = Chroma(

    persist_directory=VECTOR_DB_PATH,

    collection_name=COLLECTION_NAME,

    embedding_function=embedding_model
)



# ==================================================
# Load Documents for BM25
# ==================================================

print("Loading documents for BM25...")


stored_data = vector_db.get(

    include=[
        "documents",
        "metadatas"
    ]

)


documents = []


for text, metadata in zip(

    stored_data["documents"],

    stored_data["metadatas"]

):

    documents.append(

        Document(

            page_content=text,

            metadata=metadata

        )

    )


print(
    f"Loaded {len(documents)} documents"
)



# ==================================================
# BM25 Retriever
# ==================================================

bm25_retriever = BM25Retriever.from_documents(

    documents

)

bm25_retriever.k = 10



# ==================================================
# Vector Retriever
# ==================================================

vector_retriever = vector_db.as_retriever(

    search_type="mmr",

    search_kwargs={

        "k":10,

        "fetch_k":30

    }

)



# ==================================================
# Manual Hybrid Retrieval
# ==================================================

def hybrid_search(
    query,
    top_k=20
):

    """
    Combine BM25 + Vector results
    """


    bm25_docs = bm25_retriever.invoke(
        query
    )


    vector_docs = vector_retriever.invoke(
        query
    )


    combined = []

    seen = set()


    for doc in bm25_docs + vector_docs:


        key = doc.page_content[:200]


        if key not in seen:

            combined.append(doc)

            seen.add(key)



    return combined[:top_k]



# ==================================================
# Reranker
# ==================================================

print("Loading reranker...")


reranker = CrossEncoder(

    RERANK_MODEL

)



def rerank_documents(

    query,

    docs,

    top_k=5

):

    """
    Re-rank retrieved chunks
    """


    pairs = [

        [
            query,
            doc.page_content
        ]

        for doc in docs

    ]


    scores = reranker.predict(
        pairs
    )


    ranked = sorted(

        zip(
            docs,
            scores
        ),

        key=lambda x: x[1],

        reverse=True

    )


    results = []


    for doc, score in ranked[:top_k]:


        doc.metadata["rerank_score"] = float(score)

        results.append(doc)


    return results



# ==================================================
# Ollama LLM
# ==================================================

llm = ChatOllama(

    model="llama2",

    temperature=0,

    base_url="http://127.0.0.1:11434"

)



# ==================================================
# Memory
# ==================================================

chat_history = []

MAX_HISTORY = 10



# ==================================================
# Main RAG Function
# ==================================================

def ask_question(question):

    global chat_history



    print(
        "\n======================="
    )

    print(
        "Question:",
        question
    )



    # ----------------------------------------------
    # Retrieve
    # ----------------------------------------------

    candidates = hybrid_search(

        question,

        top_k=20

    )


    print(
        f"Hybrid retrieved: {len(candidates)}"
    )



    # ----------------------------------------------
    # Rerank
    # ----------------------------------------------

    docs = rerank_documents(

        question,

        candidates,

        top_k=5

    )


    print(
        f"After reranking: {len(docs)}"
    )



    # ----------------------------------------------
    # Build Context
    # ----------------------------------------------

    context = ""


    for i, doc in enumerate(

        docs,

        1

    ):


        context += f"""

==============================
Document {i}

Case Title:
{doc.metadata.get("title")}

Case ID:
{doc.metadata.get("case_id")}

Rerank Score:
{doc.metadata.get("rerank_score")}


Content:

{doc.page_content}

"""



    # ----------------------------------------------
    # Prompt
    # ----------------------------------------------

    prompt = f"""

Question:

{question}


Retrieved Legal Documents:

{context}


Instructions:

- Answer only from retrieved documents.
- Do not use outside knowledge.
- Mention case titles when available.
- If information is missing say:
"I don't have enough information based on the provided documents."

"""



    messages = [

        SystemMessage(

            content="""

You are a legal research assistant.
Your answers must be grounded only
in retrieved legal documents.

"""

        ),


        *chat_history[-MAX_HISTORY:],


        HumanMessage(

            content=prompt

        )

    ]



    # ----------------------------------------------
    # Generate Answer
    # ----------------------------------------------

    response = llm.invoke(

        messages

    )


    answer = response.content



    # ----------------------------------------------
    # Update Memory
    # ----------------------------------------------

    chat_history.append(

        HumanMessage(

            content=question

        )

    )


    chat_history.append(

        AIMessage(

            content=answer

        )

    )


    chat_history = chat_history[-MAX_HISTORY:]



    return answer



# ==================================================
# CLI Test
# ==================================================

def start_chat():


    print(
        "⚖️ Legal Hybrid RAG + Reranker"
    )

    print(
        "Type quit to exit"
    )


    while True:


        query = input(
            "\nYou: "
        )


        if query.lower() == "quit":

            break


        answer = ask_question(
            query
        )


        print(
            "\nAnswer:\n"
        )

        print(answer)



if __name__ == "__main__":

    start_chat()