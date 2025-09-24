# main.py - FastAPI Backend
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List
import tempfile
import os
from dotenv import load_dotenv

# Import your existing functions
from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader, PyPDFLoader
from generalUtils import summarize_chain, generate_audio
from ytUtils import get_transcript_as_document
import validators

# Load environment variables
load_dotenv()

app = FastAPI(title="Brevio API", version="1.0.0")

# Add CORS middleware to allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LLM
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)

# Pydantic models for request/response
class URLRequest(BaseModel):
    url: str
    content_type: str  # 'youtube' or 'web'

class SummaryResponse(BaseModel):
    summary: str
    audio_url: str
    success: bool
    message: str

@app.post("/summarize/url", response_model=SummaryResponse)
async def summarize_url(request: URLRequest):
    """Summarize YouTube videos or web URLs"""
    try:
        url = request.url.strip()
        
        # Validate URL
        if not validators.url(url):
            raise HTTPException(status_code=400, detail="Invalid URL format")
        
        if request.content_type == "youtube":
            if "youtube.com" not in url and "youtu.be" not in url:
                raise HTTPException(status_code=400, detail="Not a valid YouTube URL")
            
            # Process YouTube video
            docs = get_transcript_as_document(url)
            
        elif request.content_type == "web":
            if "youtube.com" in url:
                raise HTTPException(status_code=400, detail="Use YouTube endpoint for YouTube URLs")
            
            # Process web page
            loader = UnstructuredURLLoader(
                urls=[url],
                ssl_verify=True,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/115.0.0.0 Safari/537.36"
                    )
                }
            )
            docs = loader.load()
        
        else:
            raise HTTPException(status_code=400, detail="Invalid content type")
        
        # Generate summary
        summary = summarize_chain(docs, llm)
        
        # Generate audio
        audio_data, audio_b64 = generate_audio(summary)
        
        # Save audio file temporarily
        audio_filename = f"summary_{hash(url)}.mp3"
        audio_path = f"temp_audio/{audio_filename}"
        os.makedirs("temp_audio", exist_ok=True)
        
        with open(audio_path, "wb") as f:
            f.write(audio_data)
        
        return SummaryResponse(
            summary=summary,
            audio_url=f"/audio/{audio_filename}",
            success=True,
            message="Summary generated successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize/pdf", response_model=SummaryResponse)
async def summarize_pdf(files: List[UploadFile] = File(...)):
    """Summarize PDF files"""
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")
        
        documents = []
        temp_files = []
        
        # Process each uploaded file
        for uploaded_file in files:
            if not uploaded_file.filename.endswith('.pdf'):
                raise HTTPException(status_code=400, detail=f"File {uploaded_file.filename} is not a PDF")
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                content = await uploaded_file.read()
                temp_file.write(content)
                temp_path = temp_file.name
                temp_files.append(temp_path)
            
            # Load PDF
            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            documents.extend(docs)
        
        # Generate summary
        summary = summarize_chain(documents, llm)
        
        # Generate audio
        audio_data, audio_b64 = generate_audio(summary)
        
        # Save audio file
        audio_filename = f"pdf_summary_{hash(''.join([f.filename for f in files]))}.mp3"
        audio_path = f"temp_audio/{audio_filename}"
        os.makedirs("temp_audio", exist_ok=True)
        
        with open(audio_path, "wb") as f:
            f.write(audio_data)
        
        # Clean up temporary files
        for temp_path in temp_files:
            os.remove(temp_path)
        
        return SummaryResponse(
            summary=summary,
            audio_url=f"/audio/{audio_filename}",
            success=True,
            message="PDF summary generated successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    """Serve audio files"""
    audio_path = f"temp_audio/{filename}"
    if os.path.exists(audio_path):
        return FileResponse(audio_path, media_type="audio/mpeg", filename=filename)
    else:
        raise HTTPException(status_code=404, detail="Audio file not found")

@app.get("/")
async def root():
    return {"message": "Brevio API is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)