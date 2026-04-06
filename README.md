# 🎬 YouTube Q&A Assistant — AI-Powered Video Intelligence

Chat with any YouTube video like it’s a document.

Just paste a YouTube link, the app extracts the transcript, builds a **RAG (Retrieval-Augmented Generation)** knowledge base using embeddings + vector search, and lets you ask questions that are **grounded in the transcript** — reducing hallucinations and keeping answers relevant.

---

## ✨ What You Can Do

- **💬 Smart Q&A (Chat with the video)**  
  Ask anything about the content and get transcript-grounded answers.
- **📝 Auto Summary**  
  Generate a structured, comprehensive summary of the full video.
- **🎯 Key Takeaways**  
  Instantly extract the main points as bullet insights.
- **💡 Topics Extraction**  
  List the different topics discussed with brief descriptions.
- **🌍 Multi-Language Transcript Support**  
  Attempts transcripts in a preferred order (e.g., English, Hindi, Urdu, Arabic, etc.), and falls back to any available captions.
- **⚡ Semantic Search with Vector DB**  
  Uses **ChromaDB** + **MiniLM embeddings** for similarity-based retrieval.

---

## 🧠 How It Works (RAG Pipeline)

**YouTube URL → Transcript Extraction → Chunking → Embeddings → Vector Store (ChromaDB) → Retrieval → LLM Answer**

Core logic:
- Transcript is fetched from YouTube captions (including auto captions if available)
- Transcript is chunked using a recursive splitter
- Chunks are embedded with **`all-MiniLM-L6-v2`**
- ChromaDB stores embeddings locally (`./chroma_db`)
- User questions retrieve top-k relevant chunks
- The LLM answers using the retrieved transcript context

---

## 🖥️ Demo / Screenshots

Add your screenshots here (already present in the repo):

- `images/01_image.png`
- `images/02_image.png`
- `images/03_image.png`
- `images/04_image.png`

> Tip: You can embed them like:
> `![App Screenshot](images/01_image.png)`

---

## 🧰 Tech Stack

- **Streamlit** — interactive web app UI  
- **LangChain** — RAG orchestration + prompt chaining  
- **ChromaDB** — vector database for semantic retrieval  
- **HuggingFace MiniLM (`all-MiniLM-L6-v2`)** — embeddings  
- **OpenRouter** — LLM gateway (supports Gemini, GPT, Claude, Llama, etc.)
- **YouTube Transcript API** — transcript extraction

---

## ⚙️ Setup & Run Locally

### 1) Clone the repository
```bash
git clone https://github.com/AbdulRehman393/youtube-qa-assistant.git
cd youtube-qa-assistant
```

### 2) Create a virtual environment (recommended)
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

### 4) Configure environment variables

Create a `.env` file (you can copy from `.env.example`):

```bash
cp .env.example .env
```

Then set:
```env
OPENROUTER_API_KEY=your_openrouter_key_here
# Optional:
LLM_MODEL=google/gemini-2.0-flash-001
```

### 5) Run the app
```bash
streamlit run app.py
```

---

## 🔧 Configuration (Defaults)

In `config.py`:
- Chunking: `CHUNK_SIZE = 1000`, `CHUNK_OVERLAP = 200`
- Retrieval: `RETRIEVER_K = 5`
- Embeddings: `all-MiniLM-L6-v2`
- Vector store: `./chroma_db` (re-created per processed video)

---

## 🤖 Model Selection (Inside the App)

The UI allows choosing models like:
- `google/gemini-2.0-flash-001` (default)
- `openai/gpt-4o-mini`, `openai/gpt-4o`
- `anthropic/claude-3.5-sonnet`
- `meta-llama/llama-3-70b-instruct`
- `mistralai/mistral-large`

---

## 🛡️ Notes / Limitations

- Works best when the YouTube video has **captions** (manual or auto-generated).
- Private videos or videos without transcripts may fail to process.
- This project rebuilds the local Chroma DB each time a new video is processed.

---

## 📌 Project Motivation

This project is built to practice and demonstrate:
- RAG systems (retrieval + grounded generation)
- Vector embeddings + semantic search
- LLM orchestration using LangChain
- Building a clean AI product UI with Streamlit

---

## 📄 License

No license file is currently included. If you want, I can format and suggest the best license (MIT is common for projects like this).

---

## 🙌 Author

**Abdul Rehman**  
GitHub: `@AbdulRehman393`
