# streamlit_app_with_api.py
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader, PyPDFLoader
from generalUtils import summarize_chain, generate_audio
from ytUtils import get_transcript_as_document
import validators
import os
from dotenv import load_dotenv
import base64
from PIL import Image
import threading
import time

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn
from typing import List
import tempfile

# Load environment variables
load_dotenv()

# Streamlit configuration
st.set_page_config(page_title="Brevio", layout="wide")

# Original Streamlit app code (your existing code)
def run_streamlit_app():
    # Your existing Streamlit code here
    st.title("Brevio - Streamlit Version")
    st.write("This is your existing Streamlit interface")
    # ... rest of your Streamlit code

# FastAPI app for API endpoints
api_app = FastAPI(title="Brevio API")

# Add CORS middleware
api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LLM
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)

class URLRequest(BaseModel):
    url: str
    content_type: str

class SummaryResponse(BaseModel):
    summary: str
    audio_b64: str
    success: bool
    message: str

@api_app.post("/api/summarize/url")
async def api_summarize_url(request: URLRequest):
    try:
        url = request.url.strip()
        
        if not validators.url(url):
            raise HTTPException(status_code=400, detail="Invalid URL")
        
        if request.content_type == "youtube":
            docs = get_transcript_as_document(url)
        else:
            loader = UnstructuredURLLoader(
                urls=[url],
                ssl_verify=True,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
            )
            docs = loader.load()
        
        summary = summarize_chain(docs, llm)
        audio_data, audio_b64 = generate_audio(summary)
        
        return SummaryResponse(
            summary=summary,
            audio_b64=audio_b64,
            success=True,
            message="Success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_app.post("/api/summarize/pdf")
async def api_summarize_pdf(files: List[UploadFile] = File(...)):
    try:
        documents = []
        temp_files = []
        
        for uploaded_file in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                content = await uploaded_file.read()
                temp_file.write(content)
                temp_path = temp_file.name
                temp_files.append(temp_path)
            
            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            documents.extend(docs)
        
        summary = summarize_chain(documents, llm)
        audio_data, audio_b64 = generate_audio(summary)
        
        # Clean up
        for temp_path in temp_files:
            os.remove(temp_path)
        
        return SummaryResponse(
            summary=summary,
            audio_b64=audio_b64,
            success=True,
            message="Success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Function to run FastAPI in background
def run_api():
    uvicorn.run(api_app, host="0.0.0.0", port=8001)

# Main execution
if __name__ == "__main__":
    # Start FastAPI server in background thread
    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()
    
    # Give API server time to start
    time.sleep(2)
    
    # Run Streamlit app
    run_streamlit_app()
    
    # Add instructions for users
    st.sidebar.markdown("""
    ## 🚀 Modern Interface Available!
    
    The animated modern interface is available at:
    **[http://localhost:3000](http://localhost:3000)**
    
    API endpoints available at:
    - POST `/api/summarize/url`
    - POST `/api/summarize/pdf`
    """)