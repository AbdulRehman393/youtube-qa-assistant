# 🎬 YouTube Q&A Assistant — AI-Powered Video Intelligence

Paste any YouTube link and instantly **chat with the video content** — powered by a **RAG (Retrieval-Augmented Generation)** pipeline that grounds responses in the video transcript.

**No hallucinations. No guessing. Just accurate, transcript-based answers.**

---

## ✨ Highlights

- 💬 **Smart Q&A** — ask questions about any part of the video
- 📝 **Auto Summaries** — generate a well-structured full summary
- 🎯 **Key Takeaways** — extract main points as bullets
- 💡 **Topic Discovery** — list topics discussed with short descriptions
- 🌍 **Multi-language transcripts** — English, Urdu, Hindi, Arabic & more (falls back to any available captions)
- ⚡ **Semantic Search** — ChromaDB + MiniLM embeddings for fast relevant retrieval

---

## 🖼️ Screenshots

![Screenshot 1](images/01_image.png)
![Screenshot 2](images/02_image.png)
![Screenshot 3](images/03_image.png)
![Screenshot 4](images/04_image.png)

---

## 🧠 How It Works (RAG Flow)

**YouTube URL → Transcript Extraction → Chunking → Embeddings → Chroma Vector DB → Retrieval → LLM Answer**

1. The app extracts the YouTube transcript (captions / auto-captions if available).
2. Transcript is split into overlapping chunks for better retrieval.
3. Each chunk is embedded using **HuggingFace MiniLM (`all-MiniLM-L6-v2`)**.
4. Embeddings are stored in **ChromaDB** (`./chroma_db`).
5. On each question, the app retrieves the most relevant chunks (`top-k`).
6. The LLM answers using **only retrieved transcript context** (and responds honestly if info isn’t present).

---

## 🧰 Tech Stack

- **Streamlit** — UI + chat experience
- **LangChain** — RAG orchestration, prompt chaining, chat history
- **ChromaDB** — vector database for semantic retrieval
- **HuggingFace Embeddings** — `all-MiniLM-L6-v2`
- **OpenRouter** — LLM provider gateway (Gemini / GPT / Claude / Llama / etc.)
- **YouTube Transcript API** — transcript extraction

---

## 📁 Project Structure

- `app.py` — Streamlit UI (video input, quick actions, chat experience)
- `transcript_loader.py` — extracts video ID + fetches transcript (multi-language fallback)
- `rag_pipeline.py` — chunking, embeddings, ChromaDB vector store, conversational RAG chain, summarizer
- `config.py` — app + model + RAG configuration
- `requirements.txt` — dependencies
- `images/` — screenshots used in the README

---

## ⚙️ Setup (Local)

### 1) Clone the repo
```bash
git clone https://github.com/AbdulRehman393/youtube-qa-assistant.git
cd youtube-qa-assistant
```

### 2) Create & activate a virtual environment (recommended)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

### 4) Add environment variables

Create a `.env` file (you can copy from `.env.example`):
```bash
cp .env.example .env
```

Then add your OpenRouter API key:
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
# Optional:
LLM_MODEL=google/gemini-2.0-flash-001
```

### 5) Run the Streamlit app
```bash
streamlit run app.py
```

---

## 🤖 Model Selection (In-App)

From the sidebar you can switch models such as:
- `google/gemini-2.0-flash-001` (default)
- `openai/gpt-4o-mini`, `openai/gpt-4o`
- `anthropic/claude-3.5-sonnet`
- `meta-llama/llama-3-70b-instruct`
- `mistralai/mistral-large`

> The selected model updates `config.LLM_MODEL` at runtime inside the Streamlit session.

---

## 🔧 Configuration

Key defaults from `config.py`:

- **Chunk size**: `1000`
- **Chunk overlap**: `200`
- **Retriever top-k**: `5`
- **Embeddings**: `all-MiniLM-L6-v2`
- **Chroma persistence**: `./chroma_db`
- **Collection name**: `youtube_transcripts`

---

## 🛡️ Notes / Limitations

- Works best with **public YouTube videos that have captions** (manual or auto-generated).
- If a video has no transcript available, processing will fail.
- The vector DB folder (`./chroma_db`) is **recreated** when processing a new video.

---

## 🚀 LinkedIn Caption (Project Intro)

**Introducing YouTube Q&A Assistant — AI-Powered Video Intelligence.**

Paste any YouTube link and instantly chat with the video content, powered by a RAG engine that grounds every answer directly in the transcript.

No hallucinations. No guessing. Just precise, sourced answers.

**How it works:**
YouTube link → Transcript extracted → RAG engine processes it → You get accurate, grounded answers

**Key Features:**
- Smart Q&A
- Auto-Summaries
- Multi-Language
- Semantic Search (ChromaDB + MiniLM)

**Built With:** LangChain · ChromaDB · Streamlit · OpenRouter LLMs · HuggingFace MiniLM · YouTube Transcript API

---

## 📄 License

No license file is currently included.  
If you want, I can add a recommended license (commonly **MIT**) and update the README accordingly.

---

## 👤 Author

**Abdul Rehman**  
GitHub: `@AbdulRehman393`
