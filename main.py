import subprocess
import webbrowser
import time
import logging
import signal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set up logging
logging.basicConfig(filename='pipeline_log.txt', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def run_script(script):
    logging.info(f"Running {script}...")
    try:
        subprocess.run(["python", script], check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed running {script}: {e}")
        raise

def start_api():
    logging.info("Starting FastAPI server...")
    return subprocess.Popen(["uvicorn", "api:app", "--reload"])

def main():
    logging.info("Pipeline execution started.")

    # Step 1: Clean data
    run_script("data.py")

    # Step 2: Generate analytics plots
    run_script("analytics.py")

    # Step 3: Build FAISS index
    run_script("rag_qa.py")

    # Step 4: Start API
    api_process = start_api()
    time.sleep(5)  # Allow time for API to start
    webbrowser.open("http://127.0.0.1:8000/docs")

    try:
        logging.info("API server running. Press Ctrl+C to stop.")
        api_process.wait()
    except KeyboardInterrupt:
        logging.info("Shutting down FastAPI server...")
        os.kill(api_process.pid, signal.SIGTERM)
        api_process.wait()

    logging.info("Pipeline execution completed.")

if __name__ == "__main__":
    main()
