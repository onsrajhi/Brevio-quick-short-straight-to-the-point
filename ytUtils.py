from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from langchain.schema import Document
import os
from dotenv import load_dotenv
import re

load_dotenv()

def extract_youtube_video_id(url):
    """
    Extract video ID from various YouTube URL formats
    """
    # Remove any whitespace
    url = url.strip()
    
    # Handle different YouTube URL formats
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',  # Standard and short URLs
        r'(?:embed\/)([0-9A-Za-z_-]{11})',   # Embed URLs
        r'(?:watch\?v=)([0-9A-Za-z_-]{11})', # Watch URLs
        r'youtu\.be\/([0-9A-Za-z_-]{11})',   # youtu.be short URLs
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    # Fallback to original method
    if "youtube.com" in url:
        parsed = urlparse(url)
        if parsed.query:
            video_id = parse_qs(parsed.query).get('v', [None])[0]
            if video_id:
                return video_id
    elif "youtu.be" in url:
        return url.split("/")[-1].split("?")[0]
    
    return None

def get_transcript_as_document(url):
    """
    Get YouTube transcript and return as LangChain Document
    """
    video_id = extract_youtube_video_id(url)
    if not video_id:
        raise ValueError("Invalid YouTube URL - could not extract video ID")

    try:
        # Try without proxy first
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US', 'en-GB'])
        except Exception as e:
            # If proxy credentials are available, try with proxy
            proxy_username = os.getenv("proxy_username")
            proxy_password = os.getenv("proxy_password")
            
            if proxy_username and proxy_password:
                from youtube_transcript_api.proxies import WebshareProxyConfig
                proxy_config = WebshareProxyConfig(
                    proxy_username=proxy_username,
                    proxy_password=proxy_password,
                )
                transcript = YouTubeTranscriptApi.get_transcript(
                    video_id, 
                    languages=['en', 'en-US', 'en-GB'],
                    proxies=proxy_config
                )
            else:
                raise e
        
        if not transcript:
            raise RuntimeError("No transcript available for this video")
        
        # Combine all transcript entries
        full_text = "\n".join([entry["text"] for entry in transcript])
        
        if not full_text.strip():
            raise RuntimeError("Transcript is empty")
        
        return [Document(page_content=full_text, metadata={"source": url, "video_id": video_id})]
        
    except Exception as e:
        error_msg = str(e)
        if "No transcript" in error_msg or "Transcript is disabled" in error_msg:
            raise RuntimeError(f"No transcript available for this YouTube video. The video may have disabled captions or may not have auto-generated captions available.")
        elif "Video unavailable" in error_msg:
            raise RuntimeError(f"YouTube video is unavailable or private.")
        else:
            raise RuntimeError(f"Failed to fetch transcript: {error_msg}")

# Test function (optional)
def test_video_id_extraction():
    """Test function to verify video ID extraction works correctly"""
    test_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://www.youtube.com/embed/dQw4w9WgXcQ",
        "https://youtube.com/watch?v=dQw4w9WgXcQ",
    ]
    
    for url in test_urls:
        video_id = extract_youtube_video_id(url)
        print(f"URL: {url} -> ID: {video_id}")

if __name__ == "__main__":
    test_video_id_extraction()