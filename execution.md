## Execution Guide: Hotel Booking Analytics & QA System

## Prerequisites
- Python 3.10+
- Groq API key (register at https://console.groq.com/)
- Conda or virtualenv installed

## Environment Setup
conda create -n llm python=3.10 -y
conda activate llm
pip install -r requirements.txt

## Add Groq API Key
Create a .env file and paste:
GROQ_API_KEY=your_groq_api_key_here

## Run the Full Project
python main.py

## This will:
## 1. Preprocess data (data.py)
## 2. Generate plots (analytics.py)
## 3. Build vector store (rag_qa.py)
## 4. Start the FastAPI server (api.py)
## 5. Open browser to Swagger UI: http://127.0.0.1:8000/docs

## API Endpoints
- GET /           → Server status
- GET /analytics  → Booking insights + plot paths
- POST /ask       → Ask any question like:
                    - Which country had the most bookings?
                    - What is the average ADR?
 ## Output Files
- cleaned_hotel_bookings.csv  → Cleaned dataset
- plots/                      → Revenue, lead time, cancellations
- query_logs.csv              → Logs with LLM response time

## Shutdown
To stop the FastAPI server:
CTRL + C

# Done!
# You're now running a complete LLM-powered hotel booking insight system!
"""
