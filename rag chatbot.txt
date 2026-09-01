import os
import numpy as np
import faiss
import gradio as gr

from groq import Groq
from getpass import getpass
from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer


# ============================================================
# GROQ
# ============================================================

GROQ_API_KEY = getpass("Enter your NEW Groq API key: ")

client = Groq(
    api_key=GROQ_API_KEY
)

MODEL = "openai/gpt-oss-20b"


# ============================================================
# EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ============================================================
# STORAGE
# ============================================================

documents = []

chunks = []

vector_database = None


# ============================================================
# READ PDF
# ============================================================

def read_pdf(path):

    text = ""

    reader = PdfReader(path)

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# ============================================================
# READ DOCX
# ============================================================

def read_docx(path):

    doc = Document(path)

    text = ""

    for paragraph in doc.paragraphs:

        text += paragraph.text + "\n"

    return text


# ============================================================
# READ TXT
# ============================================================

def read_txt(path):

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as file:

        return file.read()


# ============================================================
# DOCUMENT READER
# ============================================================

def read_document(path):

    extension = os.path.splitext(path)[1].lower()

    if extension == ".pdf":

        return read_pdf(path)

    elif extension == ".docx":

        return read_docx(path)

    elif extension == ".txt":

        return read_txt(path)

    else:

        return ""


# ============================================================
# CHUNKING
# ============================================================

def create_chunks(
    text,
    chunk_size=700,
    overlap=100
):

    text = text.replace(
        "\n",
        " "
    )

    text = " ".join(
        text.split()
    )

    result = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():

            result.append(
                chunk.strip()
            )

        start += chunk_size - overlap

    return result


# ============================================================
# BUILD VECTOR DATABASE
# ============================================================

def build_database(files):

    global documents
    global chunks
    global vector_database

    if not files:

        return "❌ Please upload at least one document."


    documents = []

    chunks = []


    # --------------------------------------------------------
    # READ FILES
    # --------------------------------------------------------

    for file in files:

        path = file.name

        text = read_document(path)

        if not text.strip():

            continue


        file_chunks = create_chunks(
            text
        )


        for chunk in file_chunks:

            chunks.append({
                "text": chunk,
                "source": os.path.basename(path)
            })


        documents.append(
            os.path.basename(path)
        )


    if not chunks:

        return "❌ Could not extract text from the uploaded files."


    # --------------------------------------------------------
    # CREATE EMBEDDINGS
    # --------------------------------------------------------

    texts = [
        item["text"]
        for item in chunks
    ]


    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )


    embeddings = embeddings.astype(
        "float32"
    )


    # --------------------------------------------------------
    # FAISS
    # --------------------------------------------------------

    dimension = embeddings.shape[1]


    vector_database = faiss.IndexFlatIP(
        dimension
    )


    vector_database.add(
        embeddings
    )


    return (
        "✅ Knowledge base created!\n\n"
        f"Documents: {len(documents)}\n"
        f"Chunks: {len(chunks)}"
    )


# ============================================================
# RETRIEVE
# ============================================================

def retrieve(
    question,
    top_k=5
):

    if vector_database is None:

        return []


    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )


    query_embedding = query_embedding.astype(
        "float32"
    )


    scores, indexes = vector_database.search(
        query_embedding,
        min(top_k, len(chunks))
    )


    results = []


    for score, index in zip(
        scores[0],
        indexes[0]
    ):

        if index >= 0:

            results.append({

                "text":
                    chunks[index]["text"],

                "source":
                    chunks[index]["source"],

                "score":
                    float(score)

            })


    return results


# ============================================================
# RAG CHATBOT
# ============================================================

def rag_chat(
    message,
    history
):

    if not message.strip():

        return history


    # --------------------------------------------------------
    # CHECK DATABASE
    # --------------------------------------------------------

    if vector_database is None:

        answer = (
            "⚠️ Please upload your documents "
            "and click 'Build Knowledge Base' first."
        )

        history = history + [
            {
                "role": "user",
                "content": message
            },
            {
                "role": "assistant",
                "content": answer
            }
        ]

        return history


    # --------------------------------------------------------
    # RETRIEVE
    # --------------------------------------------------------

    results = retrieve(
        message,
        top_k=5
    )


    if not results:

        context = "No relevant information found."

    else:

        context_parts = []

        for result in results:

            context_parts.append(
                f"[Source: {result['source']}]\n"
                f"{result['text']}"
            )


        context = "\n\n".join(
            context_parts
        )


    # --------------------------------------------------------
    # PROMPT
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # GROQ
    # --------------------------------------------------------

    try:

        response = client.chat.completions.create(

            model=MODEL,

            messages=[
                {
                    "role": "system",
                    "content":
                    "You are a helpful RAG assistant."
                },

                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.2,

            max_tokens=4096
        )


        answer = response.choices[0].message.content


    except Exception as e:

        answer = (
            "❌ Groq error:\n\n"
            + str(e)
        )


    # --------------------------------------------------------
    # SOURCE INFORMATION
    # --------------------------------------------------------

    if results:

        sources = list(
            dict.fromkeys(
                r["source"]
                for r in results
            )
        )

        answer += (
            "\n\n---\n"
            "📚 Sources: "
            + ", ".join(sources)
        )


    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    history = history + [

        {
            "role":
                "user",

            "content":
                message
        },

        {
            "role":
                "assistant",

            "content":
                answer
        }

    ]


    return history


# ============================================================
# CLEAR CHAT
# ============================================================

def clear_chat():

    return []


# ============================================================
# GUI
# ============================================================

with gr.Blocks(
    title="MyAI RAG"
) as demo:

    gr.Markdown(
        """
# 🧠 MyAI — RAG Chatbot

Ask questions about your own documents.

**PDF • DOCX • TXT**
"""
    )


    # --------------------------------------------------------
    # FILE UPLOAD
    # --------------------------------------------------------

    files = gr.File(
        label="Upload Knowledge Base",
        file_count="multiple",
        file_types=[
            ".pdf",
            ".docx",
            ".txt"
        ]
    )


    build = gr.Button(
        "🔨 Build Knowledge Base",
        variant="primary"
    )


    status = gr.Markdown(
        "Upload documents to begin."
    )


    build.click(
        build_database,
        inputs=files,
        outputs=status
    )


    # --------------------------------------------------------
    # CHAT
    # --------------------------------------------------------

    chatbot = gr.Chatbot(
        label="MyAI",
        height=500
    )


    message = gr.Textbox(
        placeholder=
        "Ask something about your documents...",
        label="Question"
    )


    with gr.Row():

        send = gr.Button(
            "Send ➤",
            variant="primary"
        )

        clear = gr.Button(
            "🗑 Clear"
        )


    # --------------------------------------------------------
    # SEND
    # --------------------------------------------------------

    send.click(
        rag_chat,
        inputs=[
            message,
            chatbot
        ],
        outputs=[
            chatbot
        ]
    )


    message.submit(
        rag_chat,
        inputs=[
            message,
            chatbot
        ],
        outputs=[
            chatbot
        ]
    )


    # --------------------------------------------------------
    # CLEAR
    # --------------------------------------------------------

    clear.click(
        clear_chat,
        outputs=chatbot
    )


# ============================================================
# START
# ============================================================

demo.launch(
    share=True,
    debug=True
)
