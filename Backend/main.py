from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import os
import logging
from pathlib import Path

# Import your utility modules
from ytUtils import get_transcript_as_document
from generalUtils import (
    summarize_chain, 
    generate_audio, 
    load_web_content, 
    load_pdf_content,
    validate_inputs
)

# Import Groq LLM
from groq import Groq

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Brevio - AI Content Summarization API",
    description="Transform YouTube videos, web articles, and PDF documents into intelligent summaries with audio",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (for audio files)
os.makedirs("static/audio", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Groq client
try:
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    logger.info("Groq client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Groq client: {e}")
    groq_client = None

# Pydantic models
class URLRequest(BaseModel):
    url: str
    content_type: str  # "youtube" or "web"

class SummaryResponse(BaseModel):
    summary: str
    audio_url: str
    content_type: str
    source: str

# Custom LLM wrapper for Groq
class GroqLLM:
    def __init__(self, client, model_name="gemma2-9b-it"):
        self.client = client
        self.model_name = model_name
    
    def __call__(self, prompt):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2048,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise HTTPException(status_code=500, detail=f"LLM processing failed: {str(e)}")
    
    def run(self, docs):
        # For LangChain compatibility
        if isinstance(docs, list):
            text = "\n\n".join([doc.page_content for doc in docs])
        else:
            text = str(docs)
        return self.__call__(text)

# Initialize LLM
def get_llm():
    if not groq_client:
        raise HTTPException(status_code=500, detail="LLM service not available")
    return GroqLLM(groq_client)

@app.get("/")
async def root():
    return {"message": "Brevio API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "groq_available": groq_client is not None,
        "version": "2.0.0"
    }

@app.post("/summarize/url", response_model=SummaryResponse)
async def summarize_url(request: URLRequest):
    """
    Summarize content from YouTube or web URLs
    """
    try:
        logger.info(f"Processing {request.content_type} URL: {request.url}")
        
        # Validate inputs
        validate_inputs(request.content_type, url=request.url)
        
        # Load content based on type
        if request.content_type == "youtube":
            documents = get_transcript_as_document(request.url)
            logger.info("YouTube transcript loaded successfully")
        elif request.content_type == "web":
            documents = load_web_content(request.url)
            logger.info("Web content loaded successfully")
        else:
            raise HTTPException(status_code=400, detail="Invalid content type. Use 'youtube' or 'web'")
        
        if not documents:
            raise HTTPException(status_code=400, detail="No content could be extracted")
        
        # Initialize LLM
        llm = get_llm()
        
        # Generate summary
        logger.info("Generating summary...")
        summary = summarize_chain(documents, llm)
        
        if not summary or not summary.strip():
            raise HTTPException(status_code=500, detail="Failed to generate summary")
        
        # Generate audio
        logger.info("Generating audio...")
        audio_bytes, audio_b64 = generate_audio(summary)
        
        # Save audio file
        audio_filename = f"summary_{hash(request.url)}_{request.content_type}.mp3"
        audio_path = f"static/audio/{audio_filename}"
        
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)
        
        audio_url = f"/static/audio/{audio_filename}"
        
        logger.info("Processing completed successfully")
        
        return SummaryResponse(
            summary=summary,
            audio_url=audio_url,
            content_type=request.content_type,
            source=request.url
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/summarize/pdf", response_model=SummaryResponse)
async def summarize_pdf(files: List[UploadFile] = File(...)):
    """
    Summarize content from PDF files
    """
    try:
        logger.info(f"Processing {len(files)} PDF files")
        
        # Validate files
        validate_inputs("pdf", files=files)
        
        # Validate file types
        for file in files:
            if not file.filename.endswith('.pdf'):
                raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")
        
        # Load PDF content
        logger.info("Loading PDF content...")
        documents = load_pdf_content(files)
        
        if not documents:
            raise HTTPException(status_code=400, detail="No readable content found in PDF files")
        
        # Initialize LLM
        llm = get_llm()
        
        # Generate summary
        logger.info("Generating summary...")
        summary = summarize_chain(documents, llm)
        
        if not summary or not summary.strip():
            raise HTTPException(status_code=500, detail="Failed to generate summary")
        
        # Generate audio
        logger.info("Generating audio...")
        audio_bytes, audio_b64 = generate_audio(summary)
        
        # Save audio file
        filenames = "_".join([f.filename for f in files[:3]])  # Use first 3 filenames
        audio_filename = f"summary_pdf_{hash(filenames)}.mp3"
        audio_path = f"static/audio/{audio_filename}"
        
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)
        
        audio_url = f"/static/audio/{audio_filename}"
        
        logger.info("PDF processing completed successfully")
        
        return SummaryResponse(
            summary=summary,
            audio_url=audio_url,
            content_type="pdf",
            source=f"{len(files)} PDF file(s)"
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"PDF processing error: {e}")
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")

@app.get("/download/audio/{filename}")
async def download_audio(filename: str):
    """
    Download audio file
    """
    audio_path = f"static/audio/{filename}"
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        path=audio_path,
        media_type="audio/mpeg",
        filename=filename
    )

@app.delete("/cleanup")
async def cleanup_old_files():
    """
    Clean up old audio files (optional endpoint for maintenance)
    """
    try:
        audio_dir = Path("static/audio")
        deleted_count = 0
        
        for file_path in audio_dir.glob("*.mp3"):
            try:
                file_path.unlink()
                deleted_count += 1
            except Exception as e:
                logger.warning(f"Could not delete {file_path}: {e}")
        
        return {"message": f"Cleaned up {deleted_count} audio files"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    # Check if required environment variables are set
    required_env_vars = ["GROQ_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        print(f"Please set the following environment variables: {missing_vars}")
        exit(1)
    
    logger.info("Starting Brevio API server...")
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )