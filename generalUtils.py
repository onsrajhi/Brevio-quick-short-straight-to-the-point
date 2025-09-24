from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from gtts import gTTS
import base64
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from fastapi import HTTPException, UploadFile
import PyPDF2
import io
import os
from typing import List
import time
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Defining max tokens
MAX_TOKENS = 6000

def validate_inputs(content_type, url=None, files=None):
    """Validate inputs based on content type"""
    if content_type in ["youtube", "web"]:
        if not url:
            raise ValueError("URL is required")
        if not is_valid_url(url):
            raise ValueError("Invalid URL format")
    elif content_type == "pdf":
        if not files or len(files) == 0:
            raise ValueError("At least one PDF file is required")
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                raise ValueError(f"File {file.filename} is not a PDF")

def is_valid_url(url):
    """Validate URL format"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

def load_web_content(url):
    """Load content from a web URL"""
    try:
        # Add headers to mimic a real browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-us,en;q=0.5',
            'Accept-Encoding': 'gzip,deflate',
            'Connection': 'keep-alive',
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'advertisement', 'noscript']):
            element.decompose()
        
        # Try to find main content areas
        content_selectors = [
            'article',
            '[role="main"]',
            'main',
            '.content',
            '.post-content',
            '.article-content',
            '.entry-content',
            '#content',
            '.main-content',
            '.story-body',
            '.article-body',
            'div.content'
        ]
        
        content_text = ""
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                content_text = ' '.join([elem.get_text(strip=True) for elem in elements])
                if content_text:  # Only break if we got meaningful content
                    break
        
        # If no specific content area found, get all text
        if not content_text:
            content_text = soup.get_text(strip=True)
        
        # Clean up the text
        content_text = re.sub(r'\n+', '\n', content_text)
        content_text = re.sub(r'\s+', ' ', content_text)
        content_text = content_text.strip()
        
        if not content_text or len(content_text) < 100:
            raise ValueError("Unable to extract meaningful content from the webpage")
        
        logger.info(f"Successfully extracted {len(content_text)} characters from web content")
        return [Document(page_content=content_text, metadata={"source": url})]
        
    except requests.RequestException as e:
        error_msg = f"Failed to fetch web content: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    except Exception as e:
        error_msg = f"Error processing web content: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

def load_pdf_content(files: List[UploadFile]):
    """Load content from PDF files"""
    documents = []
    
    for file in files:
        try:
            logger.info(f"Processing PDF file: {file.filename}")
            
            # Read file content
            file_content = file.file.read()
            
            # Reset file pointer
            file.file.seek(0)
            
            # Create PDF reader
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            
            text_content = ""
            page_count = len(pdf_reader.pages)
            
            for page_num in range(page_count):
                try:
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + "\n"
                except Exception as page_error:
                    logger.warning(f"Error extracting text from page {page_num + 1} of {file.filename}: {page_error}")
                    continue
            
            if text_content.strip():
                # Clean up text
                text_content = re.sub(r'\n+', '\n', text_content)
                text_content = re.sub(r'\s+', ' ', text_content)
                text_content = text_content.strip()
                
                documents.append(Document(
                    page_content=text_content,
                    metadata={
                        "source": file.filename, 
                        "type": "pdf",
                        "pages": page_count,
                        "characters": len(text_content)
                    }
                ))
                logger.info(f"Successfully processed {file.filename}: {len(text_content)} characters")
            else:
                logger.warning(f"No text content extracted from {file.filename}")
                
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {str(e)}")
            continue
    
    if not documents:
        raise ValueError("No readable content found in any of the PDF files")
    
    logger.info(f"Total documents processed: {len(documents)}")
    return documents

def chunk_documents(docs):
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(docs)
    logger.info(f"Split {len(docs)} documents into {len(split_docs)} chunks")
    return split_docs

def summarize_chain(docs, llm):
    """Create summary using the provided LLM"""
    try:
        total_text = " ".join(doc.page_content for doc in docs)
        total_tokens = len(total_text) // 4  # Rough estimate: 1 token ≈ 4 characters for English
        
        logger.info(f"Processing content: {len(total_text)} characters (~{total_tokens} tokens)")

        # Use 'stuff' for summarization if under token limit else switch to 'map-reduce'
        if total_tokens < MAX_TOKENS:
            # Single-shot summarization for smaller content
            template = '''Please provide a comprehensive and well-structured summary of the following content.

Instructions:
1. Start with a clear and relevant title
2. Provide a brief introduction explaining what the content is about
3. Create main points or sections with bullet points highlighting key information
4. End with a meaningful conclusion that captures the main takeaways

Make the summary informative, easy to understand, and suitable for audio narration.

Content: {text}

Summary:'''

            combined_text = "\n\n".join([doc.page_content for doc in docs])
            
            # Use the LLM directly
            summary = llm(template.format(text=combined_text))
            logger.info("Generated summary using single-shot approach")
            return summary
            
        else:
            # For longer content, use chunking approach
            logger.info("Content is large, using chunking approach")
            chunked_docs = chunk_documents(docs)
            
            # Summarize each chunk first
            chunk_summaries = []
            for i, chunk in enumerate(chunked_docs):
                logger.info(f"Processing chunk {i+1}/{len(chunked_docs)}")
                chunk_template = '''Provide a concise summary of the key points in this text section:

{text}

Key points:'''
                chunk_summary = llm(chunk_template.format(text=chunk.page_content))
                chunk_summaries.append(chunk_summary)
            
            # Combine all chunk summaries
            combined_summaries = "\n\n".join(chunk_summaries)
            
            # Create final summary
            final_template = '''Based on these key points from different sections, create a comprehensive final summary:

{text}

Please provide:
1. A clear title
2. An introduction
3. Main points organized logically
4. A conclusion with key takeaways

Make it suitable for audio narration:'''
            
            final_summary = llm(final_template.format(text=combined_summaries))
            logger.info("Generated final summary using map-reduce approach")
            return final_summary
            
    except Exception as e:
        error_msg = f"Failed to generate summary: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

def generate_audio(summary_text, lang="en"):
    """Generate audio from text using gTTS"""
    try:
        logger.info(f"Generating audio for {len(summary_text)} characters")
        
        # Formatting text for better audio
        text = re.sub(r'[#*_>`\-]', '', summary_text)  # Remove markdown formatting
        text = re.sub(r'(?<=[^\.\!\?])\n', '. ', text)  # Add periods if line ends without one
        text = re.sub(r'\n+', ' ', text)  # Flatten newlines
        text = re.sub(r'\s{2,}', ' ', text)  # Remove extra spaces
        text = text.strip()
        
        if not text:
            raise ValueError("No text to convert to audio")
        
        # Create a unique filename
        timestamp = str(int(time.time()))
        audio_filename = f"summary_audio_{timestamp}.mp3"
        
        # Create gTTS object with better settings
        tts = gTTS(text=text, lang=lang, slow=False, tld='com')
        tts.save(audio_filename)
        
        # Read the audio file
        with open(audio_filename, "rb") as f:
            audio_bytes = f.read()
        
        # Clean up temporary file
        try:
            os.remove(audio_filename)
        except OSError as e:
            logger.warning(f"Could not remove temporary audio file: {e}")
        
        b64 = base64.b64encode(audio_bytes).decode()
        logger.info(f"Successfully generated audio: {len(audio_bytes)} bytes")
        return audio_bytes, b64
        
    except Exception as e:
        error_msg = f"Failed to generate audio: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

# Test functions for debugging
def test_web_content(url="https://httpbin.org/html"):
    """Test web content loading"""
    try:
        docs = load_web_content(url)
        print(f"Successfully loaded {len(docs)} documents")
        print(f"First 200 characters: {docs[0].page_content[:200]}...")
        return docs
    except Exception as e:
        print(f"Error loading web content: {e}")
        return None

def test_audio_generation(text="This is a test summary for audio generation."):
    """Test audio generation"""
    try:
        audio_bytes, b64 = generate_audio(text)
        print(f"Successfully generated audio: {len(audio_bytes)} bytes")
        return True
    except Exception as e:
        print(f"Error generating audio: {e}")
        return False

if __name__ == "__main__":
    # Test the functions
    print("Testing web content loading...")
    test_url = "https://httpbin.org/html"
    test_web_content(test_url)
    
    print("\nTesting audio generation...")
    test_audio_generation()