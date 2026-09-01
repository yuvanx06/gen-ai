import os
import numpy as np
import faiss
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer

# ============================================================
# CONFIG / SECRETS
# ============================================================

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

client = Groq(api_key=GROQ_API_KEY)

MODEL = "openai/gpt-oss-20b"

st.set_page_config(page_title="MyAI RAG", page_icon="🧠", layout="wide")


# ============================================================
# EMBEDDING MODEL (cached so it only loads once per session)
# ============================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()


# ============================================================
# SESSION STATE (replaces Gradio's global variables)
# ============================================================

if "documents" not in st.session_state:
    st.session_state.documents = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "vector_database" not in st.session_state:
    st.session_state.vector_database = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role": ..., "content": ...}


# ============================================================
# READ PDF / DOCX / TXT
# (Streamlit's uploaded files are file-like objects already,
#  so we read from them directly instead of a filesystem path)
# ============================================================

def read_pdf(file):
    text = ""
    reader = PdfReader(file)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def read_docx(file):
    doc = Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text


def read_txt(file):
    return file.read().decode("utf-8", errors="ignore")


def read_document(uploaded_file):
    extension = os.path.splitext(uploaded_file.name)[1].lower()

    if extension == ".pdf":
        return read_pdf(uploaded_file)
    elif extension == ".docx":
        return read_docx(uploaded_file)
    elif extension == ".txt":
        return read_txt(uploaded_file)
    else:
        return ""


# ============================================================
# CHUNKING
# ============================================================

def create_chunks(text, chunk_size=700, overlap=100):
    text = text.replace("\n", " ")
    text = " ".join(text.split())

    result = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            result.append(chunk.strip())
        start += chunk_size - overlap

    return result


# ============================================================
# BUILD VECTOR DATABASE
# ============================================================

def build_database(uploaded_files):
    if not uploaded_files:
        return "❌ Please upload at least one document."

    documents = []
    chunks = []

    for uploaded_file in uploaded_files:
        text = read_document(uploaded_file)

        if not text.strip():
            continue

        file_chunks = create_chunks(text)

        for chunk in file_chunks:
            chunks.append({
                "text": chunk,
                "source": uploaded_file.name
            })

        documents.append(uploaded_file.name)

    if not chunks:
        return "❌ Could not extract text from the uploaded files."

    texts = [item["text"] for item in chunks]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    dimension = embeddings.shape[1]
    vector_database = faiss.IndexFlatIP(dimension)
    vector_database.add(embeddings)

    # Save into session state
    st.session_state.documents = documents
    st.session_state.chunks = chunks
    st.session_state.vector_database = vector_database

    return (
        "✅ Knowledge base created!\n\n"
        f"Documents: {len(documents)}\n\n"
        f"Chunks: {len(chunks)}"
    )


# ============================================================
# RETRIEVE
# ============================================================

def retrieve(question, top_k=5):
    vector_database = st.session_state.vector_database
    chunks = st.session_state.chunks

    if vector_database is None:
        return []

    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indexes = vector_database.search(
        query_embedding,
        min(top_k, len(chunks))
    )

    results = []
    for score, index in zip(scores[0], indexes[0]):
        if index >= 0:
            results.append({
                "text": chunks[index]["text"],
                "source": chunks[index]["source"],
                "score": float(score)
            })

    return results


# ============================================================
# RAG CHAT (returns the answer string; history handled by caller)
# ============================================================

def rag_chat(message):
    if st.session_state.vector_database is None:
        return (
            "⚠️ Please upload your documents "
            "and click 'Build Knowledge Base' first."
        )

    results = retrieve(message, top_k=5)

    if not results:
        context = "No relevant information found."
    else:
        context_parts = [
            f"[Source: {r['source']}]\n{r['text']}" for r in results
        ]
        context = "\n\n".join(context_parts)

    prompt = f"""
You are a RAG-based private AI assistant.

Answer the user's question using the
provided knowledge base.

IMPORTANT RULES:

1. Use the retrieved documents as your
   primary source.

2. If the answer is not present in the
   documents, clearly say that the
   information was not found in the
   uploaded documents.

3. Do not invent facts.

4. Give a clear and useful answer.

5. At the end, mention the source
   document names used.

KNOWLEDGE BASE:

{context}

USER QUESTION:

{message}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful RAG assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=4096
        )
        answer = response.choices[0].message.content

    except Exception as e:
        answer = "❌ Groq error:\n\n" + str(e)

    if results:
        sources = list(dict.fromkeys(r["source"] for r in results))
        answer += "\n\n---\n📚 Sources: " + ", ".join(sources)

    return answer


# ============================================================
# UI — SIDEBAR: UPLOAD + BUILD KNOWLEDGE BASE
# ============================================================

st.title("🧠 MyAI — RAG Chatbot")
st.caption("Ask questions about your own documents. **PDF • DOCX • TXT**")

with st.sidebar:
    st.header("📂 Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload Knowledge Base",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    if st.button("🔨 Build Knowledge Base", type="primary"):
        with st.spinner("Building knowledge base..."):
            status_message = build_database(uploaded_files)
        st.markdown(status_message)

    if st.session_state.documents:
        st.success(f"Loaded: {', '.join(st.session_state.documents)}")

    if st.button("🗑 Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()


# ============================================================
# UI — CHAT DISPLAY
# ============================================================

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ============================================================
# UI — CHAT INPUT
# ============================================================

user_message = st.chat_input("Ask something about your documents...")

if user_message:
    st.session_state.chat_history.append(
        {"role": "user", "content": user_message}
    )
    with st.chat_message("user"):
        st.markdown(user_message)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = rag_chat(user_message)
        st.markdown(answer)

    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer}
    )
