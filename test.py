from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Extracts video ID from a YouTube URL
def extract_video_id(url):
    try:
        parsed_url = urlparse(url)
        hostname = parsed_url.hostname or ''
        if hostname == "youtu.be":
            return parsed_url.path.lstrip('/')
        if hostname in ["www.youtube.com", "youtube.com", "m.youtube.com"]:
            return parse_qs(parsed_url.query).get("v", [None])[0]
    except Exception as e:
        print(f"Error extracting video ID: {e}")
    return None

# Fetches transcript text from a YouTube video
def get_transcript_from_url(url):
    video_id = extract_video_id(url)
    if not video_id:
        print("Could not extract video ID from URL.")
        return None
    try:
        api = YouTubeTranscriptApi()
        snippets = api.fetch(video_id, languages=["en"])
        # Each snippet is a FetchedTranscriptSnippet object with .text attribute
        text = " ".join([snippet.text for snippet in snippets])
        return text
    except Exception as e:
        print(f"Error fetching transcript: {e}")
        return None

# Splits long transcript into smaller overlapping chunks for RAG
def split_text(text, chunk_size=1000, chunk_overlap=100):
    if not text:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

