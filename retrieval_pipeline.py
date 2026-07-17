from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import (SystemMessage, HumanMessage, AIMessage)


# -----------------------------------
# Configuration
# -----------------------------------

VECTOR_DB_PATH = "db/chroma_db"
COLLECTION_NAME = "caselaw_rag"


# -----------------------------------
# Load Embedding Model
# -----------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={
        "device": "cpu"
    },
    encode_kwargs={
        "normalize_embeddings": True
    }
)



# -----------------------------------
# Connect Chroma Vector DB
# -----------------------------------

db = Chroma(
    persist_directory=VECTOR_DB_PATH,
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_model
)


# -----------------------------------
# Local LLM
# -----------------------------------

model = ChatOllama(
    model="llama2",
    temperature=0,
    base_url="http://127.0.0.1:11434"
)



# -----------------------------------
# Retriever
# -----------------------------------

retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 20
    }
)


# -----------------------------------
# Conversation Memory
# -----------------------------------

chat_history = []

MAX_HISTORY = 10

# -----------------------------------
# Ask Question
# -----------------------------------

def ask_question(question):

    global chat_history

    print(f"\n--- Question ---\n{question}")

    # -------------------------------
    # Step 1: Rewrite question
    # -------------------------------

    search_question = question

    if chat_history:
        rewrite_messages = [

            SystemMessage(
                content=(
                    "Rewrite the user's question into a standalone "
                    "legal research question using the conversation history. "
                    "Return only the rewritten question."
                )
            )

        ] + chat_history + [HumanMessage(content=question)]

        try:
            result = model.invoke(rewrite_messages)

            search_question = (result.content.strip())

            print(f"\nSearch query: {search_question}")

        except Exception as e:
            print(f"Question rewrite failed: {e}")


    # -------------------------------
    # Step 2: Retrieve documents
    # -------------------------------

    docs = retriever.invoke(search_question)

    print(f"\nRetrieved {len(docs)} legal chunks")


    for i, doc in enumerate(docs, 1):
        print(f"\n[{i}] {doc.metadata.get('title')}")

        print(doc.page_content[:200], "...")


    # -------------------------------
    # Step 3: Build Context
    # -------------------------------

    context_parts = []

    for i, doc in enumerate(docs, 1):
        metadata = doc.metadata

        context_parts.append(f"""[Document {i}]
                                Case Title:{metadata.get('title')}
                                Case ID:{metadata.get('case_id')}
                                Content:{doc.page_content}
                            """)

        context = "\n\n".join(context_parts)

    # -------------------------------
    # Step 4: Generate Answer
    # -------------------------------

    prompt = f"""Question:{question}
        Legal Documents:{context}
        Instructions:
        - Answer ONLY from the provided legal documents.
        - Do not use outside knowledge.
        - If the documents do not contain the answer, say:
        "I don't have enough information to answer that question based on the provided documents."
        - Mention relevant case titles when possible.
        """.strip()

    messages = [
        SystemMessage(
            content=(
                "You are a legal research RAG assistant. "
                "Your answers must be grounded only in retrieved cases."
            )
        ),
        *chat_history[-MAX_HISTORY:],

            HumanMessage(
                content=prompt
            )
    ]

    response = model.invoke(messages)

    answer = response.content


    # -------------------------------
    # Step 5: Update Memory
    # -------------------------------

    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=answer))
    chat_history = chat_history[-MAX_HISTORY:]

    print("\nAnswer:\n")
    print(answer)

    return answer

# -----------------------------------
# Chat Loop
# -----------------------------------

def start_chat():
    print("⚖️ Legal RAG Assistant ready!")

    print("Type 'quit' to exit.")

    while True:
        question = input("\nYou: ")

        if question.lower() == "quit":
            print("Goodbye!")
            break

        ask_question(question)

# -----------------------------------
# Entry Point
# -----------------------------------

if __name__ == "__main__":

    start_chat()