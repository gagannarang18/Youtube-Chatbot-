import os
import streamlit as st
from backend.transcript_utils import get_transcript_from_url
from backend.rag_chain import RAGChain
from backend.rag_utils import format_answer

# ── STREAMLIT PAGE SETUP ────────────────────────────────────────────────────────
st.set_page_config(page_title="YouTube ChatBot 💬", layout="centered")
st.title("🎥 YouTube ChatBot")
st.markdown("Ask questions about a YouTube video using Bedrock embeddings + Groq LLM.")

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

youtube_url = st.text_input("📺 Enter YouTube video URL:")

if youtube_url:
    # 1) Fetch the transcript
    with st.spinner("📄 Fetching transcript..."):
        transcript = get_transcript_from_url(youtube_url)

    if not transcript:
        st.error("❌ Could not fetch transcript from that URL.")
        st.stop()

    # 2) Initialize RAGChain and load or build the FAISS store
    rag = RAGChain()
    # build_or_load: if `faiss_index/` exists, it simply loads;
    # otherwise it builds from `transcript` and saves to disk
    rag.load_or_build(path="faiss_index", text=transcript)

    # 3) Build the QA chain
    qa_chain = rag.build_chain()

    # 4) Ask the user for a query
    query = st.text_input("💬 Ask a question about the video:")

    if query:
        with st.spinner("🧠 Thinking..."):
            # Use run() or __call__ with dict depending on your RetrievalQA API
            result = qa_chain({"query": query})
        # 5) Format & display
        st.markdown(format_answer(result))
