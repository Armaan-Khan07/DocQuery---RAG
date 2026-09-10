"""
DocQuery — Chat with Your Documents (RAG)
Upload PDFs/txt files, ask questions, get grounded answers with citations.
"""

import streamlit as st
from dotenv import load_dotenv
from src.ingest import process_uploaded_file
from src.vectorstore import VectorStore
from src.generate import generate_answer

load_dotenv()

st.set_page_config(page_title="DocQuery — RAG Assistant", page_icon="📄", layout="wide")

# ---------- Session state ----------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------- Sidebar: upload & settings ----------
with st.sidebar:
    st.title("📄 DocQuery")
    st.caption("RAG-powered document Q&A assistant")

    st.divider()
    st.subheader("1. Upload documents")
    uploaded_files = st.file_uploader(
        "PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True
    )

    if st.button("Process documents", type="primary", disabled=not uploaded_files):
        with st.spinner("Extracting, chunking, and embedding..."):
            all_chunks = []
            for f in uploaded_files:
                if f.name not in st.session_state.processed_files:
                    chunks = process_uploaded_file(f)
                    all_chunks.extend(chunks)
                    st.session_state.processed_files.append(f.name)

            if all_chunks:
                if st.session_state.vectorstore is None:
                    st.session_state.vectorstore = VectorStore()
                    st.session_state.vectorstore.build(all_chunks)
                else:
                    st.session_state.vectorstore.add(all_chunks)
                st.success(f"Indexed {len(all_chunks)} chunks from {len(uploaded_files)} file(s).")
            else:
                st.info("Files already processed.")

    if st.session_state.processed_files:
        st.divider()
        st.subheader("Indexed files")
        for fname in st.session_state.processed_files:
            st.markdown(f"- {fname}")

    st.divider()
    st.subheader("2. Retrieval settings")
    search_mode = st.radio("Search mode", ["Hybrid (semantic + keyword)", "Semantic only"])
    top_k = st.slider("Chunks to retrieve", 1, 10, 4)

    if st.button("Clear all"):
        st.session_state.vectorstore = None
        st.session_state.processed_files = []
        st.session_state.chat_history = []
        st.rerun()

# ---------- Main: chat interface ----------
st.header("Chat with your documents")

if st.session_state.vectorstore is None:
    st.info("👈 Upload and process at least one document to get started.")
else:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("View retrieved sources"):
                    for s in msg["sources"]:
                        st.markdown(f"**{s['source']}** (score: {s['score']:.3f})")
                        st.caption(s["text"][:300] + "...")

    query = st.chat_input("Ask a question about your documents...")

    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant chunks..."):
                vs = st.session_state.vectorstore
                if search_mode.startswith("Hybrid"):
                    results = vs.hybrid_search(query, top_k=top_k)
                else:
                    results = vs.semantic_search(query, top_k=top_k)

            with st.spinner("Generating answer..."):
                answer = generate_answer(query, results)

            st.markdown(answer)
            with st.expander("View retrieved sources"):
                for s in results:
                    st.markdown(f"**{s['source']}** (score: {s['score']:.3f})")
                    st.caption(s["text"][:300] + "...")

        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer, "sources": results}
        )