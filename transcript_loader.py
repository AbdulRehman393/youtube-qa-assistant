from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from requests import Session
import urllib3

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def extract_video_id(url: str) -> str:
    """
    Extract video ID from various YouTube URL formats.
    Supports:
        - https://www.youtube.com/watch?v=VIDEO_ID
        - https://youtu.be/VIDEO_ID
        - https://www.youtube.com/embed/VIDEO_ID
    """
    parsed_url = urlparse(url)

    if parsed_url.hostname in ("www.youtube.com", "youtube.com"):
        if parsed_url.path == "/watch":
            return parse_qs(parsed_url.query).get("v", [None])[0]
        elif parsed_url.path.startswith("/embed/"):
            return parsed_url.path.split("/embed/")[1]
    elif parsed_url.hostname == "youtu.be":
        return parsed_url.path.lstrip("/")

    return None


def get_transcript(video_url: str) -> dict:
    """
    Fetch transcript from a YouTube video URL.
    Supports ALL languages (Hindi, Urdu, English, etc.)
    """
    video_id = extract_video_id(video_url)

    if not video_id:
        raise ValueError(
            "Invalid YouTube URL. Please provide a valid YouTube video link."
        )

    try:
        # Create a custom session that skips SSL verification
        session = Session()
        session.verify = False

        # Pass session as http_client to YouTubeTranscriptApi
        ytt_api = YouTubeTranscriptApi(http_client=session)

        # First, list all available transcripts for this video
        transcript_list = ytt_api.list(video_id)

        # Try to find transcript in preferred order: en, hi, ur, then any available
        preferred_languages = ["en", "hi", "ur", "ar", "es", "fr", "de", "pt", "ja", "ko", "zh"]
        
        fetched_transcript = None
        language_found = None

        # Try preferred languages first
        for lang in preferred_languages:
            try:
                fetched_transcript = transcript_list.find_transcript([lang]).fetch()
                language_found = lang
                break
            except Exception:
                continue

        # If none of the preferred languages found, grab whatever is available
        if fetched_transcript is None:
            for transcript in transcript_list:
                fetched_transcript = transcript.fetch()
                language_found = transcript.language_code
                break

        if fetched_transcript is None:
            raise Exception("No transcripts available for this video.")

        # Convert to raw data (list of dicts)
        raw_data = fetched_transcript.to_raw_data()

        # Combine all segments into a single text
        full_text = " ".join(segment["text"] for segment in raw_data)

        # Keep segments with timestamps for reference
        segments = [
            {
                "text": segment["text"],
                "start": segment["start"],
                "duration": segment["duration"],
            }
            for segment in raw_data
        ]

        return {
            "text": full_text,
            "segments": segments,
            "video_id": video_id,
            "language": language_found,
        }

    except Exception as e:
        raise Exception(f"Error fetching transcript: {str(e)}")


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"