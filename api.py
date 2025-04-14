from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from groq import Groq
from dotenv import load_dotenv
import time
import csv
from datetime import datetime

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY in environment variables.")

groq_client = Groq(api_key=GROQ_API_KEY)

app = FastAPI(
    title="Hotel Booking Analyst API",
    description="Ask questions about hotel booking dataset using RAG + Groq LLM",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MODEL_NAME = "llama3-8b-8192"
CSV_PATH = "cleaned_hotel_bookings.csv"

# Load data
df = pd.read_csv(CSV_PATH)
chunks = [
    f"""
    Hotel: {row['hotel']}
    Arrival Date: {row['arrival_date']}
    Country: {row['country']}
    Lead Time: {row['lead_time']}
    ADR: {row['adr']}
    Nights: {row['total_nights']}
    Canceled: {row['is_canceled']}
    """.strip()
    for _, row in df.iterrows()
]

model = SentenceTransformer(EMBEDDING_MODEL)
embeddings = model.encode(chunks)
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings, dtype='float32'))

def retrieve(query, k=10):
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec, dtype='float32'), k)
    return [chunks[i] for i in I[0]]

def ask_llm(context, query):
    prompt = f"""
You are a hotel booking analyst. Use the context below to answer the question clearly and concisely.

Context:
{context}

Question: {query}
"""
    start = time.time()
    response = groq_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    end = time.time()
    latency_ms = round((end - start) * 1000, 2)
    answer = response.choices[0].message.content
    
    # Logging to CSV
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "answer": answer,
        "latency_ms": latency_ms
    }
    log_path = "query_logs.csv"
    file_exists = os.path.isfile(log_path)
    with open(log_path, "a", newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=log_entry.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_entry)

    return answer, latency_ms

class QueryInput(BaseModel):
    query: str

@app.get("/")
def root():
    return {"message": "API is live. Use /docs to try it!"}

@app.post("/ask")
def answer_question(payload: QueryInput):
    user_query = payload.query
    context_docs = retrieve(user_query, k=10)
    retrieved = "\n---\n".join(context_docs)

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
    answer, latency = ask_llm(context, user_query)
    return {"question": user_query, "answer": answer, "latency_ms": latency}

@app.get("/analytics")
def get_analytics():
    total_bookings = len(df)
    canceled = df['is_canceled'].sum()
    cancel_rate = 100 * canceled / total_bookings
    avg_adr = df['adr'].mean()
    avg_lead_time = df['lead_time'].mean()
    top_country = df['country'].value_counts().idxmax()

    return {
        "total_bookings": total_bookings,
        "cancellations": int(canceled),
        "cancellation_rate": round(cancel_rate, 2),
        "average_adr": round(avg_adr, 2),
        "average_lead_time": round(avg_lead_time, 2),
        "top_country": top_country,
        "plots": [
            "plots/country_distribution.png",
            "plots/lead_time_distribution.png",
            "plots/cancellation_rate.png",
            "plots/revenue_timeline.png"
        ]
    }
