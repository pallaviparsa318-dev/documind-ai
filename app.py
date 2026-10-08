"""Run with: streamlit run app.py"""
import requests
import streamlit as st
from documind.core import (
    CHAT_MODEL, EMBED_MODEL, OLLAMA_URL, answer_question,
    collection_at, ingest_pdf, retrieve,
)

st.set_page_config(page_title="DocuMind AI", page_icon="📚", layout="wide")
st.title("📚 DocuMind AI")
st.caption("Local PDF question answering with semantic retrieval and page-level source references")

@st.cache_resource
def get_collection():
    return collection_at()

collection = get_collection()
with st.sidebar:
    st.header("System")
    st.write(f"**Chat:** `{CHAT_MODEL}`")
    st.write(f"**Embeddings:** `{EMBED_MODEL}`")
    try:
        health = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        health.raise_for_status()
        available = {m["name"].split(":")[0] for m in health.json().get("models", [])}
        missing = [m for m in [CHAT_MODEL, EMBED_MODEL] if m.split(":")[0] not in available]
        if missing:
            st.warning("Pull missing models: " + ", ".join(missing))
        else:
            st.success("Ollama is running")
    except requests.RequestException:
        st.error("Ollama not reachable. Start Ollama before asking questions.")
    st.metric("Indexed chunks", collection.count())
    if st.button("Clear document index", type="secondary"):
        collection.delete(ids=collection.get()["ids"]) if collection.count() else None
        st.session_state.pop("history", None)
        st.rerun()

st.subheader("1. Upload documents")
uploads = st.file_uploader("Choose one or more text-based PDF files", type="pdf", accept_multiple_files=True)
if uploads and st.button("Index PDFs", type="primary"):
    for file in uploads:
        try:
            with st.spinner(f"Indexing {file.name}..."):
                count = ingest_pdf(file.getvalue(), file.name, collection)
            if count:
                st.success(f"Indexed {file.name}: {count} chunks")
            else:
                st.warning(f"No selectable text found in {file.name}. Scanned PDFs need OCR.")
        except Exception as exc:
            st.error(f"Could not index {file.name}: {exc}")
    st.rerun()

st.subheader("2. Ask a question")
if "history" not in st.session_state:
    st.session_state.history = []
for item in st.session_state.history:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])
        if item.get("sources"):
            with st.expander("Retrieved source passages"):
                for i, p in enumerate(item["sources"], 1):
                    st.markdown(f"**Source {i}: {p['source']} — page {p['page']}**")
                    st.caption(p["text"])
question = st.chat_input("Ask something about your uploaded documents...")
if question:
    st.session_state.history.append({"role": "user", "content": question})
    try:
        with st.spinner("Retrieving and answering..."):
            passages = retrieve(question, collection)
            response = answer_question(question, passages)
        st.session_state.history.append({"role": "assistant", "content": response, "sources": passages})
    except Exception as exc:
        st.session_state.history.append({"role": "assistant", "content": f"Error: {exc}"})
    st.rerun()

st.divider()
st.caption("Privacy: Documents and vector index stay on this machine. Do not expose this development app publicly without authentication. Generated answers can be wrong; verify source excerpts.")
