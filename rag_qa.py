# rag_qa.py

import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from groq import Groq
from dotenv import load_dotenv

# ---- CONFIG ----
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CSV_PATH = "cleaned_hotel_bookings.csv"
INDEX_FILE = "faiss_index.bin"
MODEL_NAME = "llama3-8b-8192"
CHUNK_SIZE = 300

# Initialize Groq client
load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---- 1. Load & Chunk Data ----
def chunk_text(df):
    chunks = []
    for _, row in df.iterrows():
        base = f"""
        Hotel: {row['hotel']}
        Arrival Date: {row['arrival_date']}
        Country: {row['country']}
        Lead Time: {row['lead_time']}
        ADR: {row['adr']}
        Nights: {row['total_nights']}
        Canceled: {row['is_canceled']}
        """
        chunks.append(base.strip())
    return chunks

# ---- 2. Embed & Build Vector Index ----
def build_faiss_index(chunks):
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype='float32'))
    return index, embeddings

# ---- 3. Retrieve Top-K Relevant Chunks ----
def retrieve(query, index, chunks, k=10):
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec, dtype='float32'), k)
    return [chunks[i] for i in I[0]]

# ---- 4. Ask Groq LLM ----
def ask_llm(context, query):
    prompt = f"""
You are a hotel booking analyst. Use the context below to answer the question clearly and concisely.

Context:
{context}

Question: {query}
"""
    response = groq_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content

# ---- 5. Main ----
if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    chunks = chunk_text(df)
    index, _ = build_faiss_index(chunks)
    faiss.write_index(index, INDEX_FILE)

    user_query = "Which country had the most bookings?"
    context_docs = retrieve(user_query, index, chunks, k=10)
    retrieved = "\n---\n".join(context_docs)

    # Add computed statistics to context
    top_country = df['country'].value_counts().idxmax()
    top_count = df['country'].value_counts().max()
    avg_lead = df['lead_time'].mean()
    avg_adr = df['adr'].mean()
    cancel_rate = 100 * df['is_canceled'].sum() / len(df)

    context = f"""
Booking Summary:
- Most booked country: {top_country} with {top_count} bookings
- Average lead time: {avg_lead:.2f} days
- Average ADR: €{avg_adr:.2f}
- Cancellation rate: {cancel_rate:.2f}%

Sample Booking Records:
{retrieved}
"""

    answer = ask_llm(context, user_query)

    print("\nQuestion:", user_query)
    print("Answer:", answer)
