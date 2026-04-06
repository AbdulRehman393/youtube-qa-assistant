import os
from dotenv import load_dotenv

load_dotenv()

# App Configuration
APP_NAME = "YouTube Q&A Assistant"
APP_ICON = "🎬"

# OpenRouter Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = os.getenv("LLM_MODEL", "google/gemini-2.0-flash-001")

# ChromaDB Configuration
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "youtube_transcripts"

# Text Splitting Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retriever Configuration
RETRIEVER_K = 5

# Embedding Model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"