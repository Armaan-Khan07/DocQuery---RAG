# DocQuery — RAG-based Document Q&A Assistant

A Retrieval-Augmented Generation (RAG) app that lets you upload PDFs/text files and ask
questions answered directly from your documents — with source citations, not hallucinations.

## Why this project
Most portfolio "chatbots" just wrap an LLM API. This one demonstrates the full RAG pipeline:
chunking strategy, embedding, hybrid retrieval (semantic + keyword), and grounded generation —
the actual skills companies look for in AI/ML and applied LLM roles.

## Architecture
```
Upload (PDF/TXT)
      │
      ▼
  ingest.py      → extract text, clean, chunk (overlapping, boundary-aware)
      │
      ▼
 vectorstore.py  → embed chunks (MiniLM), store in FAISS
      │             + BM25 index for keyword matching
      ▼
   Query  →  hybrid_search()  →  top-k relevant chunks
      │
      ▼
  generate.py    → build grounded prompt → Groq LLM (Llama 3.1) → cited answer
      │
      ▼
  app.py (Streamlit UI) → chat interface with source inspection
```

## Setup
```bash
cd docquery-rag
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and add your free Groq API key from https://console.groq.com

streamlit run app.py
```

## Features
- **PDF & TXT ingestion** with page-aware, boundary-aware chunking
- **Hybrid search**: combines semantic similarity (embeddings) + BM25 keyword matching
- **Grounded generation**: LLM answers only from retrieved context, cites sources
- **Multi-document support**: upload and query across several files at once
- **Transparent retrieval**: expandable panel shows exactly which chunks were used

## Tech stack
| Layer       | Choice                          | Why |
|-------------|----------------------------------|-----|
| Embeddings  | `sentence-transformers` (MiniLM) | Free, local, fast, no API cost |
| Vector DB   | FAISS                            | Industry standard, in-memory, simple |
| Keyword     | BM25 (`rank-bm25`)               | Catches exact-term matches embeddings miss |
| LLM         | Groq (Llama 3.1 8B)               | Free tier, very low latency |
| UI          | Streamlit                        | Fast to build, good for demos |

## Possible extensions (for later)
- Swap FAISS for a persistent DB (Chroma/Pinecone) so the index survives restarts
- Add a cross-encoder re-ranker on top of hybrid search results
- Add RAGAS evaluation (faithfulness, answer relevancy) to quantify quality
- Support DOCX/CSV ingestion
- Add conversation memory so follow-up questions use chat history in retrieval

## Portfolio framing
Pitch this as an **"Enterprise Knowledge Assistant"** — the same architecture powers
internal HR-policy bots, technical-manual assistants, and customer-support tools companies
actually deploy. Emphasize the hybrid retrieval choice and the "no hallucination" grounding
in interviews — that's the part that shows you understand *why* RAG exists, not just how to
call an API.
