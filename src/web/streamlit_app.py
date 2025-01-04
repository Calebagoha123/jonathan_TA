import os

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
from src.rag.retriever import RAGHandler
from dotenv import load_dotenv

load_dotenv()

# Initialize RAG Handler
@st.cache_resource
def get_rag_handler():
    return RAGHandler()

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "👋 Hello! I'm Jonathan, your CSSci course assistant. Ask me anything about your courses, assignments, or deadlines!"}
        ]

def main():
    st.set_page_config(
        page_title="Jonathan CSSci Assistant",
        page_icon="📚",
        layout="centered"
    )

    st.title("📚 Ask Jonathan")

    # Initialize session state
    initialize_session_state()
    
    # Initialize RAG handler
    rag_handler = get_rag_handler()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask your question here..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Get response from RAG
        try:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = rag_handler.chat(prompt)
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Error: {str(e)}")

    # Add sidebar with information
    with st.sidebar:
        st.title("About")
        st.markdown("""
        This course assistant can help you with:
        - Assignment deadlines
        - Course content questions
        - Grading policies
        - Administrative queries
        
        All responses are based on official course materials from canvas.
        """)
        
        # Add New Chat button
        if st.button("New Chat"):
            st.session_state.messages = []
            st.rerun()  # Force streamlit to rerun the app

    if "messages" not in st.session_state:
        st.session_state.messages = []

if __name__ == "__main__":
    main()