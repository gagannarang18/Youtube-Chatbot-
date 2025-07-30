import streamlit as st
from backend.transcript_utils import get_transcript_from_url
from backend.rag_chain import RAGChain
from backend.rag_utils import format_answer

# Setup page
st.set_page_config(page_title="YouTube ChatBot 💬", layout="centered")
st.title("🎥 YouTube ChatBot")
st.markdown("Ask questions about a YouTube video using Bedrock + Groq LLM.")

# Input: YouTube URL
youtube_url = st.text_input("📺 Enter YouTube video URL:")

if youtube_url:
    with st.spinner("📄 Fetching transcript..."):
        transcript = get_transcript_from_url(youtube_url)

    if transcript:
        # Initialize RAG only once
        rag = RAGChain()
        rag.load_vectorstore("faiss_index")  # Ensure it's already built from transcript
        qa_chain = rag.build_chain()

        query = st.text_input("💬 Ask a question about the video:")
        if query:
            with st.spinner("🧠 Thinking..."):
                result = qa_chain.run(query)
                st.markdown(format_answer({"result": result}))
    else:
        st.error("❌ Could not fetch transcript.")
