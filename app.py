import streamlit as st
from backend.transcript_utils import get_transcript_from_url
from backend.rag_chain import run_rag_pipeline

st.set_page_config(page_title="YouTube ChatBot 💬", layout="centered")
st.title("🎥 YouTube ChatBot")
st.markdown("Ask questions about a YouTube video using Bedrock + Groq LLM.")

# Input: YouTube URL
youtube_url = st.text_input("📺 Enter YouTube video URL:")

if youtube_url:
    with st.spinner("📄 Fetching transcript..."):
        transcript = get_transcript_from_url(youtube_url)

    if transcript:
        query = st.text_input("💬 Ask a question about the video:")
        if query:
            with st.spinner("🧠 Thinking..."):
                result = run_rag_pipeline(transcript, query)
                st.success(result)
    else:
        st.error("❌ Could not fetch transcript.")
