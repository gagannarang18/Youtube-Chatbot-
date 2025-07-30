from youtube_transcript_api import YouTubeTranscriptApi     
from urllib.parse import urlparse, parse_qs

def extract_video_id(url):
    try:
        parsed_url = urlparse(url)
        if parsed_url.hostname in ["youtu.be"]:
            return parsed_url.path[1:]
        if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
            return parse_qs(parsed_url.query).get("v", [None])[0]
    except Exception:
        return None
    
def get_transcript_from_url(url):
    video_id = extract_video_id(url)
    if not video_id:
        return None
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([entry["text"] for entry in transcript])
        return text
    except Exception as e:
        print(f"Error fetching transcript: {e}")
        return None
    