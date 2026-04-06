import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from transcript_loader import get_transcript
from rag_pipeline import (
    split_transcript,
    create_vectorstore,
    get_conversational_chain,
    get_summary,
)
import config


# ─── Page Configuration ───
st.set_page_config(
    page_title=config.APP_NAME,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Netflix Premium Dark UI ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    .stApp {
        background: #0a0a0a !important;
        font-family: 'Inter', sans-serif;
        color: #f1f1f1 !important;
    }

    .stApp p, .stApp li, .stApp span, .stApp div, .stApp label,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: #f1f1f1 !important;
    }

    .stMarkdown p {
        color: #f1f1f1 !important;
        font-size: 0.95rem !important;
        line-height: 1.8 !important;
    }

    .stMarkdown li {
        color: #e8e8e8 !important;
        font-size: 0.93rem !important;
        line-height: 1.8 !important;
    }

    .stMarkdown strong {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .hero-container {
        text-align: center;
        padding: 2rem 1rem 1rem 1rem;
    }

    .hero-badge {
        display: inline-block;
        background: rgba(229, 9, 20, 0.12);
        border: 1px solid rgba(229, 9, 20, 0.3);
        color: #ff4d56 !important;
        padding: 0.35rem 1rem;
        border-radius: 50px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 900;
        color: #ffffff !important;
        margin: 0.5rem 0;
        line-height: 1.1;
        letter-spacing: -1px;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #b0b0b0 !important;
        font-weight: 400;
        margin-top: 0.5rem;
        line-height: 1.6;
    }

    .hero-accent {
        color: #ff4d56 !important;
        font-weight: 700;
    }

    .glow-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(229, 9, 20, 0.5), transparent);
        margin: 1.5rem auto;
        max-width: 600px;
        border: none;
    }

    .section-label {
        color: #ff4d56 !important;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    .section-title {
        color: #ffffff !important;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }

    .section-desc {
        color: #999999 !important;
        font-size: 0.85rem;
        margin-bottom: 1rem;
    }

    .stTextInput > div > div > input {
        background-color: #141414 !important;
        border: 2px solid #333333 !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        padding: 0.8rem 1.2rem !important;
        font-size: 1rem !important;
        font-family: 'Inter', sans-serif !important;
        caret-color: #ff4d56 !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: #e50914 !important;
        box-shadow: 0 0 15px rgba(229, 9, 20, 0.15) !important;
        background-color: #1a1a1a !important;
    }

    .stTextInput > div > div > input::placeholder {
        color: #666666 !important;
        -webkit-text-fill-color: #666666 !important;
    }

    input, textarea {
        background-color: #141414 !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #e50914, #b20710) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.7rem 2rem !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 4px 15px rgba(229, 9, 20, 0.3) !important;
        font-family: 'Inter', sans-serif !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #ff1a25, #e50914) !important;
        box-shadow: 0 6px 25px rgba(229, 9, 20, 0.5) !important;
        transform: translateY(-2px) !important;
    }

    .action-card {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        margin-bottom: 0.5rem;
    }

    .action-card:hover {
        border-color: #e50914;
        box-shadow: 0 8px 30px rgba(229, 9, 20, 0.15);
        transform: translateY(-3px);
    }

    .action-icon { font-size: 2rem; margin-bottom: 0.5rem; }
    .action-title { color: #ffffff !important; font-size: 0.95rem; font-weight: 700; margin-bottom: 0.3rem; }
    .action-desc { color: #999999 !important; font-size: 0.75rem; }

    .stats-bar {
        display: flex;
        gap: 1.5rem;
        justify-content: center;
        margin: 1.5rem 0;
        flex-wrap: wrap;
    }

    .stat-item {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 12px;
        padding: 0.8rem 1.5rem;
        text-align: center;
        min-width: 120px;
    }

    .stat-value { color: #ff4d56 !important; font-size: 1.4rem; font-weight: 800; }
    .stat-label { color: #999999 !important; font-size: 0.7rem; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; margin-top: 0.2rem; }

    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin: 2rem 0;
    }

    .feature-card {
        background: #1a1a1a;
        border: 1px solid #222;
        border-radius: 14px;
        padding: 1.8rem 1.2rem;
        text-align: center;
        transition: all 0.3s ease;
    }

    .feature-card:hover {
        border-color: #e50914;
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(229, 9, 20, 0.1);
    }

    .feature-icon { font-size: 2.2rem; margin-bottom: 0.8rem; }
    .feature-title { color: #ffffff !important; font-size: 0.95rem; font-weight: 700; margin-bottom: 0.4rem; }
    .feature-desc { color: #999999 !important; font-size: 0.78rem; line-height: 1.5; }

    .stChatMessage {
        background: #1a1a1a !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 14px !important;
        padding: 1.2rem 1.4rem !important;
        margin-bottom: 0.8rem !important;
    }

    [data-testid="stChatMessageContent"] p {
        color: #f1f1f1 !important;
        font-size: 0.95rem !important;
        line-height: 1.85 !important;
    }

    [data-testid="stChatMessageContent"] li {
        color: #e8e8e8 !important;
        font-size: 0.93rem !important;
        line-height: 1.85 !important;
    }

    [data-testid="stChatMessageContent"] strong,
    [data-testid="stChatMessageContent"] b {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    [data-testid="stChatMessageContent"] h1,
    [data-testid="stChatMessageContent"] h2,
    [data-testid="stChatMessageContent"] h3,
    [data-testid="stChatMessageContent"] h4 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    [data-testid="stChatMessageContent"] code {
        background: #2a2a2a !important;
        color: #f8f8f2 !important;
        padding: 0.15rem 0.5rem !important;
        border-radius: 6px !important;
    }

    [data-testid="stChatMessageContent"] a {
        color: #ff6b73 !important;
    }

    [data-testid="stChatMessageContent"] blockquote {
        border-left: 3px solid #e50914 !important;
        padding-left: 1rem !important;
        color: #cccccc !important;
    }

    .stChatInput > div {
        background-color: #141414 !important;
        border: 2px solid #333333 !important;
        border-radius: 14px !important;
    }

    .stChatInput > div:focus-within {
        border-color: #e50914 !important;
        box-shadow: 0 0 15px rgba(229, 9, 20, 0.12) !important;
        background-color: #1a1a1a !important;
    }

    .stChatInput textarea {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        font-family: 'Inter', sans-serif !important;
        background-color: transparent !important;
        caret-color: #ff4d56 !important;
    }

    .stChatInput textarea::placeholder {
        color: #666666 !important;
        -webkit-text-fill-color: #666666 !important;
    }

    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] div {
        background-color: #141414 !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    [data-testid="stSidebar"] {
        background: #0d0d0d !important;
        border-right: 1px solid #1e1e1e !important;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div {
        color: #d0d0d0 !important;
    }

    [data-testid="stSidebar"] .stSelectbox > div > div {
        background-color: #1a1a1a !important;
        border: 1px solid #2a2a2a !important;
        color: #ffffff !important;
        border-radius: 10px !important;
    }

    [data-testid="stStatusWidget"] {
        background: #1a1a1a !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 14px !important;
    }

    [data-testid="stStatusWidget"] p,
    [data-testid="stStatusWidget"] span,
    [data-testid="stStatusWidget"] div {
        color: #e0e0e0 !important;
    }

    .stSpinner > div > div { border-top-color: #e50914 !important; }

    hr { border-color: #1e1e1e !important; }

    .app-footer {
        text-align: center;
        padding: 2rem 0;
        margin-top: 2rem;
        border-top: 1px solid #1a1a1a;
    }

    .app-footer p { color: #555555 !important; font-size: 0.75rem; }
    .app-footer a { color: #ff4d56 !important; text-decoration: none; }

    @media (max-width: 768px) {
        .hero-title { font-size: 2rem; }
        .feature-grid { grid-template-columns: 1fr; }
        .stats-bar { flex-direction: column; align-items: center; }
    }
</style>
""", unsafe_allow_html=True)


# ─── Initialize Session State ───
for key, default in {
    "messages": [], "chain": None, "retriever": None,
    "transcript_text": None, "video_processed": False,
    "video_id": None, "chat_history": [],
    "transcript_language": None, "num_segments": 0, "num_chunks": 0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ─── Sidebar ───
with st.sidebar:
    st.markdown('<p style="color:#ff4d56!important;font-size:0.7rem;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;">SETTINGS</p>', unsafe_allow_html=True)

    if config.OPENROUTER_API_KEY:
        st.success("🔑 API Key Loaded")
    else:
        st.error("⚠️ Add API Key to .env file")
        st.code("OPENROUTER_API_KEY=your-key-here", language="text")

    st.divider()

    model = st.selectbox("🤖 AI Model", [
        "google/gemini-2.0-flash-001", "openai/gpt-4o-mini",
        "openai/gpt-4o", "anthropic/claude-3.5-sonnet",
        "meta-llama/llama-3-70b-instruct", "mistralai/mistral-large",
    ], index=0)
    config.LLM_MODEL = model

    st.divider()

    if st.button("🔄 Reset Everything", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ─── HERO ───
if not st.session_state.video_processed:
    st.markdown(f"""
    <div class="hero-container">
        <div class="hero-badge">✦ AI-POWERED VIDEO INTELLIGENCE</div>
        <div class="hero-title">{config.APP_ICON} {config.APP_NAME}</div>
        <div class="hero-subtitle">
            Drop any YouTube link and <span class="hero-accent">instantly chat</span> with the video.<br>
            Summarize, extract insights, and ask questions — powered by AI.
        </div>
    </div>
    <div class="glow-divider"></div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <div class="feature-title">Smart Q&A</div>
            <div class="feature-desc">Ask any question and get accurate answers from the video content</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📝</div>
            <div class="feature-title">Auto Summary</div>
            <div class="feature-desc">Get comprehensive summaries of hours-long videos in seconds</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🌍</div>
            <div class="feature-title">Multi-Language</div>
            <div class="feature-desc">Supports Hindi, Urdu, English, Arabic & many more languages</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── URL INPUT ───
st.markdown("""
<div class="section-label">▶ VIDEO INPUT</div>
<div class="section-title">Paste your YouTube URL</div>
<div class="section-desc">Works with any public YouTube video that has captions or auto-generated subtitles</div>
""", unsafe_allow_html=True)

video_url = st.text_input("URL", placeholder="https://www.youtube.com/watch?v=...", label_visibility="collapsed")

_, center_col, _ = st.columns([1, 1, 1])
with center_col:
    process_btn = st.button("🚀 Process Video", use_container_width=True, type="primary")


# ─── PROCESS ───
if process_btn and video_url:
    if not config.OPENROUTER_API_KEY:
        st.error("⚠️ Please add your OpenRouter API Key to the .env file!")
    else:
        with st.status("🎬 Processing your video...", expanded=True) as status:
            st.write("📥 Fetching transcript...")
            try:
                result = get_transcript(video_url)
                st.session_state.transcript_text = result["text"]
                st.session_state.video_id = result["video_id"]
                st.session_state.transcript_language = result.get("language", "unknown")
                st.session_state.num_segments = len(result["segments"])
                st.write(f"✅ Found {len(result['segments'])} segments ({result.get('language', 'N/A')})")
            except Exception as e:
                st.error(f"❌ {str(e)}")
                st.stop()

            st.write("🔨 Chunking transcript...")
            documents = split_transcript(result["text"])
            st.session_state.num_chunks = len(documents)
            st.write(f"✅ {len(documents)} chunks created")

            st.write("🧠 Building knowledge base...")
            vectorstore = create_vectorstore(documents)
            st.write("✅ Vector store ready")

            st.write("⛓️ Connecting AI...")
            chain, retriever = get_conversational_chain(vectorstore)
            st.session_state.chain = chain
            st.session_state.retriever = retriever
            st.session_state.video_processed = True
            st.session_state.messages = []
            st.session_state.chat_history = []
            status.update(label="✅ Ready to chat!", state="complete", expanded=False)


# ─── POST-PROCESSING ───
if st.session_state.video_processed and st.session_state.video_id:

    lang = st.session_state.transcript_language
    st.markdown(f"""
    <div class="stats-bar">
        <div class="stat-item">
            <div class="stat-value">{st.session_state.num_segments}</div>
            <div class="stat-label">Segments</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{st.session_state.num_chunks}</div>
            <div class="stat-label">Chunks</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{lang.upper() if lang else "?"}</div>
            <div class="stat-label">Language</div>
        </div>
        <div class="stat-item">
            <div class="stat-value" style="color:#22c55e!important;">●</div>
            <div class="stat-label" style="color:#22c55e!important;">Ready</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_v1, col_v2, col_v3 = st.columns([1, 2, 1])
    with col_v2:
        st.video(f"https://www.youtube.com/watch?v={st.session_state.video_id}")

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;margin:1rem 0;">
        <div class="section-label">⚡ QUICK ACTIONS</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="action-card"><div class="action-icon">📝</div><div class="action-title">Summarize</div><div class="action-desc">Full video summary</div></div>', unsafe_allow_html=True)
        if st.button("📝 Summarize", use_container_width=True, key="sum"):
            with st.spinner("✍️ Writing summary..."):
                summary = get_summary(st.session_state.transcript_text)
                st.session_state.messages.append({"role": "user", "content": "Summarize this video"})
                st.session_state.messages.append({"role": "assistant", "content": summary})
                st.rerun()

    with col2:
        st.markdown('<div class="action-card"><div class="action-icon">🎯</div><div class="action-title">Takeaways</div><div class="action-desc">Key points</div></div>', unsafe_allow_html=True)
        if st.button("🎯 Takeaways", use_container_width=True, key="take"):
            if st.session_state.chain:
                with st.spinner("🔍 Extracting..."):
                    answer = st.session_state.chain.invoke({"question": "What are the key takeaways and main points discussed? List them as bullet points.", "chat_history": st.session_state.chat_history})
                    st.session_state.chat_history.append(HumanMessage(content="Key takeaways?"))
                    st.session_state.chat_history.append(AIMessage(content=answer))
                    st.session_state.messages.append({"role": "user", "content": "What are the key takeaways?"})
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.rerun()

    with col3:
        st.markdown('<div class="action-card"><div class="action-icon">💡</div><div class="action-title">Topics</div><div class="action-desc">What\'s discussed</div></div>', unsafe_allow_html=True)
        if st.button("💡 Topics", use_container_width=True, key="topics"):
            if st.session_state.chain:
                with st.spinner("🔍 Finding topics..."):
                    answer = st.session_state.chain.invoke({"question": "What are all the different topics discussed in this video? List each topic with a brief description.", "chat_history": st.session_state.chat_history})
                    st.session_state.chat_history.append(HumanMessage(content="Topics discussed?"))
                    st.session_state.chat_history.append(AIMessage(content=answer))
                    st.session_state.messages.append({"role": "user", "content": "What topics are discussed?"})
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.rerun()


# ─── CHAT ───
st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-label">💬 CONVERSATION</div>', unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask anything about the video..."):
    if not st.session_state.video_processed:
        st.warning("⚠️ Please process a YouTube video first!")
    elif not st.session_state.chain:
        st.error("⚠️ Something went wrong. Please reset and try again.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.chain.invoke({"question": prompt, "chat_history": st.session_state.chat_history})
                st.session_state.chat_history.append(HumanMessage(content=prompt))
                st.session_state.chat_history.append(AIMessage(content=answer))
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})


# ─── FOOTER ───
st.markdown("""
<div class="app-footer">
    <p>Built with ❤️ using <a href="https://streamlit.io">Streamlit</a> · <a href="https://openrouter.ai">OpenRouter</a> · <a href="https://python.langchain.com">LangChain</a></p>
</div>
""", unsafe_allow_html=True)