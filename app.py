import gradio as gr
import pdfplumber
import os
import json
import re
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from transformers import pipeline
from unidecode import unidecode

# -----------------------
# Load Embedding Model
# -----------------------
embed_model = SentenceTransformer("BAAI/bge-small-en")

# -----------------------
# Load LLM (small model)
# -----------------------
llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=256,
    device_map="auto"
)

# -----------------------
# Chroma DB (persistent)
# -----------------------
CHROMA_DIR = "vector_store"
os.makedirs(CHROMA_DIR, exist_ok=True)

client = chromadb.Client(Settings(persist_directory=CHROMA_DIR))
collection = client.get_or_create_collection("rag_collection")

# -----------------------
# Text Cleaning
# -----------------------
def clean_text(t):
    if not t: return ""
    t = unidecode(t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = t.strip()
    return t

# -----------------------
# PDF → Raw Text
# -----------------------
def extract_pdf_text(pdf_file):
    pages = []
    with pdfplumber.open(pdf_file) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            raw = page.extract_text() or ""
            pages.append({"page": i, "text": clean_text(raw)})
    return pages

# -----------------------
# Chunking (simple)
# -----------------------
def chunk_text(text, chunk_size=250, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# -----------------------
# Build Prompt
# -----------------------
def build_prompt(query, chunks):
    context = "\n\n".join([c["text"] for c in chunks])
    prompt = f"""
Use ONLY the context below to answer. If the answer is not present, say:
"The answer is not available in the document."

### Context:
{context}

### Question:
{query}

### Answer:
"""
    return prompt

# -----------------------
# RAG Pipeline
# -----------------------
def process_pdf(pdf):
    # clear collection
    try:
        client.delete_collection("rag_collection")
    except:
        pass

    global collection
    collection = client.get_or_create_collection("rag_collection")

    pages = extract_pdf_text(pdf)
    
    docs = []
    ids = []
    metas = []
    
    chunk_id = 0
    for page in pages:
        chunks = chunk_text(page["text"])
        for c in chunks:
            ids.append(str(chunk_id))
            docs.append(c)
            metas.append({"page": page["page"]})
            chunk_id += 1

    # embed and store
    embeddings = embed_model.encode(docs).tolist()

    collection.add(
        ids=ids,
        documents=docs,
        embeddings=embeddings,
        metadatas=metas
    )

    return f"Indexed {len(docs)} chunks from PDF."

def answer_query(query):
    # retrieve
    q_emb = embed_model.encode([query]).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=5)

    if len(res["documents"][0]) == 0:
        return "No relevant context found."

    chunks = []
    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
        chunks.append({"text": doc, "page": meta["page"]})

    # build prompt
    prompt = build_prompt(query, chunks)

    # generate
    ans = llm(prompt)[0]["generated_text"]

    # return with citation
    cited = "\n".join([f"Page {c['page']}: {c['text'][:120]}..." for c in chunks])

    return f"{ans}\n\n\nSources:\n{cited}"

# -----------------------
# Gradio Interface
# -----------------------
with gr.Blocks() as demo:
    gr.Markdown("# 📘 Chat with Your PDF (RAG System)")
    
    with gr.Tab("Upload & Index PDF"):
        pdf_input = gr.File(label="Upload PDF")
        index_btn = gr.Button("Process PDF")
        index_output = gr.Textbox(label="Status")
        
        index_btn.click(process_pdf, inputs=pdf_input, outputs=index_output)

    with gr.Tab("Ask Questions"):
        query = gr.Textbox(label="Ask a question")
        answer_btn = gr.Button("Get Answer")
        output = gr.Textbox(label="Answer")
        
        answer_btn.click(answer_query, inputs=query, outputs=output)

demo.launch()
