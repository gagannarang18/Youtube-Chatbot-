from langchain_groq import ChatGroq
import streamlit as st

def get_llm():
    return ChatGroq(
        temperature=0.2,
        model="llama-3.3-70b-versatile",
        api_key=st.secrets["GROQ_API_KEY"]
    )
