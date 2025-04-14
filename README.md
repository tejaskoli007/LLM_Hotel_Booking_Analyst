# LLM_Hotel_Booking_Analyst
A complete Retrieval-Augmented Generation (RAG) project using Groq LLM to answer analytical queries on hotel booking data. Includes data preprocessing, visualization, vector search with FAISS, and a FastAPI interface.
# LLM-Powered Hotel Booking Analytics & Question Answering System

This project was developed for the Solvei8 AI/ML Internship. It analyzes hotel booking data and allows natural language questions using FAISS + Groq LLM.

---

## Project Structure
 # LLM_Booking_Analyst/
 # ├──  data.py # Cleans and processes raw data 
 # ├── analytics.py # Creates visual plots and summaries 
 # ├── rag_qa.py # Builds vector index and answers questions 
 # ├── api.py # FastAPI server with /ask and /analytics 
 # ├── main.py # Runs the entire pipeline end-to-end 
 # ├── requirements.txt # Python dependencies 
 # ├── query_logs.csv # Logged queries and response times 
 # ├── readme.md # Project documentation 
 # └── plots/ # Visual output files


---

## How to Run

1. Install Python 3.10
2. Create and activate environment:

conda create -n llm python=3.10 -y
conda activate llm
pip install -r requirements.txt

## Add your Groq API key to a file named .env(if not found, create one):
GROQ_API_KEY=your_key_here

#Run the project:
python main.py

## API Endpoints
GET / → Server health

POST /ask → Ask questions about the dataset

GET /analytics → View booking stats and plot file paths

# Output
plots/: Contains visual PNGs

query_logs.csv: Stores queries and LLM response time

Swagger UI: http://127.0.0.1:8000/docs

# Example Questions

Which country had the most bookings?

What is the average ADR?

How long is the average stay?
