import streamlit as st
import pandas as pd

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from retrieval_pipeline import ask_question


# --------------------------------------------------
# Page Config
# --------------------------------------------------

st.set_page_config(
    page_title="Legal Case RAG Assistant",
    page_icon="⚖️",
    layout="wide"
)


st.title("⚖️ Legal Case RAG Assistant")


# --------------------------------------------------
# Configuration
# --------------------------------------------------

VECTOR_DB_PATH = "db/chroma_db"
COLLECTION_NAME = "caselaw_rag"


# --------------------------------------------------
# Load Vector Store
# --------------------------------------------------

@st.cache_resource
def load_vectorstore():

    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )


    vectorstore = Chroma(
        persist_directory=VECTOR_DB_PATH,
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model
    )


    return vectorstore



try:

    vectorstore = load_vectorstore()

    collection = vectorstore._collection

    st.success("✅ Legal ChromaDB loaded successfully")


    # --------------------------------------------------
    # Database Statistics
    # --------------------------------------------------

    total_chunks = collection.count()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Chunks", total_chunks)

    with col2:
        st.metric("Collection", COLLECTION_NAME)

    with col3:
        st.metric("Embedding", "BGE-base")

    st.divider()


    # --------------------------------------------------
    # Vector DB Preview
    # --------------------------------------------------

    st.header("📄 Legal Vector Database Preview")

    preview_limit = st.slider(
        "Number of chunks to display",
        min_value=1,
        max_value=min(100, total_chunks)
        if total_chunks > 0 else 1,
        value=min(10, total_chunks)
        if total_chunks > 0 else 1
    )


    results = collection.get(
        limit=preview_limit,
        include=[
            "documents",
            "metadatas"
        ]
    )

    if results["documents"]:
        rows = []

        for i, (doc, metadata) in enumerate(
            zip(
                results["documents"],
                results["metadatas"]
            )
        ):

            rows.append({
                "Chunk": i + 1,
                "Case Title":
                    metadata.get(
                        "title",
                        "Unknown"
                    ),
                "Case ID":
                    metadata.get(
                        "case_id",
                        "Unknown"
                    ),
                "Length":
                    len(doc),
                "Preview":
                    (
                        doc[:200] + "..."
                        if len(doc) > 200
                        else doc
                    )
            })


        df = pd.DataFrame(rows)

        st.dataframe(df, use_container_width=True, hide_index=True)


        # ----------------------------------------------
        # Full Chunk Viewer
        # ----------------------------------------------

        st.subheader("📑 View Legal Case Chunk")


        selected_chunk = st.selectbox(
            "Select Chunk",
            options=list(range(1, len(results["documents"]) + 1))
        )


        index = selected_chunk - 1

        st.write("### Metadata")

        st.json(results["metadatas"][index])

        st.write("### Case Text")

        st.text_area(
            "Document Content",
            value=results["documents"][index],
            height=400
        )

    else:
        st.warning("No chunks found in vector database.")

    st.divider()


    # --------------------------------------------------
    # Chat Interface
    # --------------------------------------------------

    st.header("💬 Ask Legal Questions")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    prompt = st.chat_input("Ask about case law...")


    if prompt:
        st.session_state.messages.append({
                "role": "user",
                "content": prompt
            }
        )


        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Searching cases and generating response..."):
            answer = ask_question(prompt)

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            }
        )

except Exception as e:
    st.error(f"❌ Error: {str(e)}")

    st.info("""
            Check:
            ✅ ChromaDB exists:
            `db/chroma_db`

            ✅ Ingestion completed:
            `python ingestion.py`

            ✅ Collection name:
            `caselaw_rag`

            ✅ Embedding model matches:
            `BAAI/bge-base-en-v1.5`

            ✅ Ollama is running:
            `ollama serve`

            ✅ Model installed:
            `ollama list`
                    """
        )