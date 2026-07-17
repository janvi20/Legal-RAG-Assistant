# ⚖️ Legal RAG Assistant

A production-style **Retrieval-Augmented Generation (RAG)** application for searching and answering questions from legal case documents.

The system ingests legal case data from the **Common Pile CaseLaw Access Project**, creates semantic embeddings, stores them in a vector database, performs **hybrid retrieval (BM25 + vector similarity)**, applies **cross-encoder reranking**, and generates grounded answers using a local LLM.

---

## 🌟 Project Highlights

### Status: In-progress

- End-to-end legal document RAG pipeline
- Streaming dataset ingestion from Hugging Face
- Hybrid search architecture:
  - Sparse retrieval (BM25)
  - Dense retrieval (Vector embeddings)
- Cross-encoder reranking
- Persistent vector database
- Local LLM inference
- Interactive Streamlit interface
- Docker-ready architecture

---

# 🏗️ System Architecture

```
                         User
                           |
                           v

                    Streamlit UI
                           |
                           v

                 RAG Application Layer
                           |
                           v

              Query Understanding Layer
                           |
                           v

          +----------------+----------------+
          |                                 |
          v                                 v

    BM25 Retriever                 Vector Retriever
    Keyword Search                 BGE Embeddings

          |                                 |
          +----------------+----------------+
                           |
                           v

                 Hybrid Retrieval
              (Candidate Documents)

                           |
                           v

              Cross Encoder Reranker
             (BAAI/bge-reranker-base)

                           |
                           v

                 Top Relevant Cases

                           |
                           v

                    Ollama LLM

                           |
                           v

              Grounded Legal Answer
```

---

# 🔄 Data Ingestion Pipeline

```
Common Pile CaseLaw Dataset
            |
            v
   Streaming Data Loader
            |
            v
    Document Processing
            |
            v
       Text Chunking
            |
            v
 BAAI/bge-base-en-v1.5 Embeddings
            |
            v
        ChromaDB
```

---

# 🧰 Technology Stack

| Layer | Technology |
|---|---|
| Programming Language | Python |
| RAG Framework | LangChain |
| Dataset | Common Pile CaseLaw Access Project |
| UI | Streamlit |
| Embedding Model | BAAI/bge-base-en-v1.5 |
| Reranker Model | BAAI/bge-reranker-base |
| Vector Database | ChromaDB |
| Sparse Retrieval | BM25 |
| LLM Runtime | Ollama |

---


## Create Virtual Environment

```bash
python -m venv .venv
```

Activate environment:

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📥 Dataset Ingestion

The project uses Hugging Face streaming to avoid downloading the complete dataset.

Run:

```bash
python ingest.py
```

The ingestion pipeline:

1. Streams legal cases
2. Extracts document content
3. Creates chunks
4. Generates embeddings
5. Stores vectors in ChromaDB


Output:

```
docs/
 └── cases_50.json


db/
 └── chroma_db/
```

---

# 🔎 Retrieval Strategy

The system uses a multi-stage retrieval architecture.

## Stage 1: Hybrid Retrieval

### BM25 Retrieval

Used for:

- Exact legal terms
- Case names
- Citations
- Statutes


Example:

```
"Miranda v Arizona"
```

---

### Vector Similarity Search

Used for:

- Semantic understanding
- Similar legal concepts
- Natural language questions


Example:

```
When can a person be held liable for another person's injuries?
```

---

## Stage 2: Reranking

Retrieved documents are reranked using:

```
BAAI/bge-reranker-base
```

Pipeline:

```
Hybrid Retriever

      |
      v

20 Candidate Documents

      |
      v

Cross Encoder

      |
      v

Top 5 Relevant Documents
```

---

# 🤖 Local LLM Setup

Install Ollama:

```
https://ollama.com
```

Download model:

```bash
ollama pull llama2
```

Start Ollama server:

```bash
ollama serve
```

The RAG pipeline communicates with:

```
http://127.0.0.1:11434
```

---

# ▶️ Running Application

## Start Streamlit

```bash
streamlit run main.py
```

Open:

```
http://localhost:8501
```
---


# 🧪 Example Queries

Try:

```
What cases discuss breach of contract?
```

```
Explain the court reasoning in negligence cases.
```

```
What factors determine legal liability?
```

```
Summarize the precedent established by these cases.
```

```
Find cases involving constitutional rights.
```

---

# 📈 Evaluation

Future evaluation framework:

## Retrieval Metrics

- Precision@K
- Recall@K
- MRR
- NDCG


## Generation Metrics

- Answer Faithfulness
- Context Relevance
- Hallucination Detection
- RAGAS Evaluation

---
 
✅ Local LLM inference  

## Future Improvements

- Move ChromaDB → Qdrant/Milvus
- Add metadata filtering
- Add document access control
- Add RAG evaluation pipeline
- Add API authentication
- Add Kubernetes deployment
- Add CI/CD automation
- Add monitoring with Prometheus/Grafana

---

# 🏆 Engineering Skills Demonstrated

This project demonstrates:

- Retrieval-Augmented Generation system design
- Information retrieval engineering
- Vector database implementation
- NLP embedding systems
- Cross encoder reranking
- Local LLM deployment

---

# ⚖️ Disclaimer

This project is for educational and research purposes only.

The generated responses are not legal advice and should not replace consultation with a qualified legal professional.
