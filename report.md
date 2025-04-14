# Implementation Report

## Objective
Build a Retrieval-Augmented Generation (RAG) system that can answer analytical queries about hotel booking data using a Groq-hosted LLM and vector search.

## Implementation Overview

### Modules
- **data.py**: Cleans the raw dataset and adds engineered features like `total_guests`, `total_nights`, and combines date columns.
- **analytics.py**: Generates visual insights such as cancellation rates, revenue trends, and geo distributions.
- **rag_qa.py**: Implements vector search using SentenceTransformer and FAISS. Uses Groq API (llama3-8b-8192) to answer questions based on retrieved context.
- **api.py**: FastAPI application that serves the QA model. Accepts queries and responds with LLM-generated answers.
- **main.py**: End-to-end orchestrator that runs preprocessing, analytics, indexing, and launches the FastAPI server.

### Stack
- **Language:** Python 3.12
- **LLM:** `llama3-8b-8192` via Groq API
- **Vector DB:** FAISS
- **Embedding Model:** `all-MiniLM-L6-v2` from SentenceTransformers
- **Framework:** FastAPI for API interaction
- **Visualization:** Matplotlib, Seaborn
- **Environment Management:** `.env` and `requirements.txt`

## Challenges
- **Large File Sizes**: FAISS index was ~175MB, which exceeded GitHub's 100MB limit. Added it to `.gitignore` and opted for dynamic rebuilding during runtime.
- **Data Size**: 119K+ records required optimized chunking and retrieval logic.
- **LLM Token Limits**: Ensured compact, informative prompts to avoid token overflow during LLM queries.

## Achievements
-  Seamless integration of Groq LLM with fast vector retrieval.
-  Modular and extensible codebase.
-  Visual analytics support.
-  Full local API server for natural language querying
