# Chat-with-PDF — RAG Assistant

- Upload any PDF → Ask questions → Get accurate, source-grounded answers
- Built with SentenceTransformers, ChromaDB, Transformers, and Gradio.

# Live Demo

👉 Hugging Face Space: [Click here](https://huggingface.co/spaces/Phantom611/Rag-pdf-assistant)


# Overview

This project implements an end-to-end Retrieval-Augmented Generation (RAG) system that turns any PDF into an interactive AI assistant.

Users can upload a PDF, automatically index its content, and ask questions — receiving answers grounded strictly in the document with citations.

This is the same architecture used by:

ChatGPT Retrieval Plugins

Google’s Enterprise Search

Notion AI Q&A

ChatPDF

# ✨ Features
## ✔ Upload & Index Any PDF

### Automatic:

- text extraction

- cleaning

- chunking

- embedding

- vector storage

## ✔ Semantic Search via ChromaDB

Powered by BAAI/BGE-small-en embeddings
Fast and highly accurate retrieval.

## ✔ LLM-Generated Answers with Citations

LLM uses ONLY retrieved context.
Reduces hallucinations and forces grounding.

## ✔ Full Web App UI (Gradio)

Live, hosted, and publicly shareable via HuggingFace Spaces.

## ✔ End-to-End RAG Pipeline

- PDF → text

- chunking

- embeddings

- vector search

- LLM answer generation

- cite pages used
