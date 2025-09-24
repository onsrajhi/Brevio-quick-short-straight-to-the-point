from urllib.parse import urlparse, parse_qs
from langchain.docstore.document import Document
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from gtts import gTTS
import os
from dotenv import load_dotenv
import re

load_dotenv()

def extract_youtube_video_id(url):
    url = url.strip()
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'youtu\.be\/([0-9A-Za-z_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    parsed = urlparse(url)
    if parsed.query:
        video_id = parse_qs(parsed.query).get('v', [None])[0]
        if video_id:
            return video_id
    return None

def get_transcript_as_document(url):
    video_id = extract_youtube_video_id(url)
    if not video_id:
        raise ValueError("Invalid YouTube URL - could not extract video ID")
    
    transcript = None
    language_codes = ['en', 'en-US', 'en-GB', 'en-CA', 'en-AU']
    for lang in language_codes:
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
            break
        except:
            continue
    if not transcript:
        try:
            transcript = next(iter(YouTubeTranscriptApi.list_transcripts(video_id))).fetch()
        except Exception as e:
            raise RuntimeError(f"No transcript available: {e}")
    
    full_text = " ".join([t["text"] for t in transcript])
    full_text = re.sub(r'\s+', ' ', full_text).strip()
    
    return [Document(page_content=full_text, metadata={"source": url, "video_id": video_id})]

# Tests
def test_video_id_extraction():
    test_urls = [
        "https://youtu.be/HZ4j_U3FC94?si=2d81p9VHOwd2A3IX",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://www.youtube.com/embed/dQw4w9WgXcQ",
        "https://youtube.com/watch?v=dQw4w9WgXcQ",
    ]
    for url in test_urls:
        print(f"URL: {url} -> ID: {extract_youtube_video_id(url)}")

def test_transcript_fetch(video_id="HZ4j_U3FC94"):
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        docs = get_transcript_as_document(url)
        print(f"Transcript fetched successfully ({len(docs[0].page_content)} chars)")
        print(docs[0].page_content[:500], "...")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("Testing video ID extraction...")
    test_video_id_extraction()
    print("\nTesting transcript fetch...")
    test_transcript_fetch()
