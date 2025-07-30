import os
import streamlit as st
from backend.transcript_utils import get_transcript_from_url
from backend.rag_chain import RAGChain
from backend.rag_utils import format_answer

# ── STREAMLIT PAGE SETUP ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YouTube ChatBot 💬", 
    layout="centered",
    page_icon="▶️",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .stTextInput input {
            border-radius: 20px;
            padding: 10px 15px;
        }
        .stButton button {
            border-radius: 20px;
            background-color: #ff4b4b;
            color: white;
            padding: 10px 24px;
        }
        .stSpinner div {
            text-align: center;
        }
        .video-info {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .answer-box {
            background-color: #f0f2f6;
            border-radius: 15px;
            padding: 20px;
            margin-top: 20px;
            border-left: 5px solid #ff4b4b;
        }
        .sidebar .sidebar-content {
            background-color: #ffffff;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar with additional info
with st.sidebar:
    st.title("ℹ️ About")
    st.markdown("""
    **YouTube ChatBot** lets you:
    - Ask questions about any YouTube video
    - Get instant answers powered by AI
    - Understand video content without watching
    """)
    st.markdown("---")
    st.markdown("🔍 **How it works:**")
    st.markdown("1. Paste a YouTube URL")
    st.markdown("2. The system fetches the transcript")
    st.markdown("3. Ask any question about the video")
    st.markdown("4. Get accurate answers instantly")
    st.markdown("---")
    st.markdown("🛠 **Technologies used:**")
    st.markdown("- Bedrock embeddings")
    st.markdown("- Groq LLM")
    st.markdown("- FAISS vector store")
    st.markdown("- RAG architecture")

# Main content
col1, col2 = st.columns([1, 3])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/1384/1384060.png", width=100)
with col2:
    st.title("YouTube ChatBot")
    st.markdown("Ask questions about any YouTube video with AI-powered understanding")

# Divider
st.markdown("---")

# ── HELPERS ─────────────────────────────────────────────────────────────────────
def ensure_secrets():
    """Set environment variables from Streamlit secrets."""
    required_keys = [
        "AWS_DEFAULT_REGION",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "GROQ_API_KEY",
    ]

    for key in required_keys:
        if key in st.secrets:
            os.environ.setdefault(key, st.secrets[key])

# ── MAIN APP ────────────────────────────────────────────────────────────────────
ensure_secrets()

# YouTube URL input with better visual
with st.container():
    st.subheader("📺 Enter YouTube Video URL")
    youtube_url = st.text_input(
        "Paste the YouTube video URL here and press Enter", 
        label_visibility="collapsed",
        placeholder="https://www.youtube.com/watch?v=..."
    )

if youtube_url:
    # 1) Fetch the transcript
    with st.spinner("📄 Fetching transcript... This may take a moment depending on video length."):
        transcript = get_transcript_from_url(youtube_url)

    if not transcript:
        st.error("❌ Could not fetch transcript from that URL. Please try a different video.")
        st.stop()
    
    # Show success message
    st.success("✅ Successfully loaded video transcript!")
    
    # Video info section
    with st.container():
        st.subheader("🎥 Video Ready for Questions")
        st.markdown("""
        You can now ask any question about this video's content.
        The AI will analyze the transcript and provide accurate answers.
        """)

    # 2) Initialize RAGChain and load or build the FAISS store
    with st.spinner("⚙️ Initializing AI engine..."):
        rag = RAGChain()
        # build_or_load: if `faiss_index/` exists, it simply loads;
        # otherwise it builds from `transcript` and saves to disk
        rag.load_or_build(path="faiss_index", text=transcript)

    # 3) Build the QA chain
    qa_chain = rag.build_chain()

    # 4) Ask the user for a query
    st.subheader("💬 Ask Your Question")
    query = st.text_input(
        "Type your question about the video content and press Enter", 
        label_visibility="collapsed",
        placeholder="What was the main point of the video?",
        key="query_input"
    )

    if query:
        with st.spinner("🧠 Analyzing video content and generating answer..."):
            # Use run() or __call__ with dict depending on your RetrievalQA API
            result = qa_chain({"query": query})
        
        # 5) Format & display
        st.subheader("📝 Answer")
        with st.container():
            st.markdown(f'<div class="answer-box">{format_answer(result)}</div>', unsafe_allow_html=True)
        
        # Feedback section
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col2:
            st.markdown("Was this answer helpful?")
            thumbs_up = st.button("👍 Yes")
            thumbs_down = st.button("👎 No")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em;">
        YouTube ChatBot • Powered by Bedrock and Groq • AI-powered video understanding
    </div>
""", unsafe_allow_html=True)