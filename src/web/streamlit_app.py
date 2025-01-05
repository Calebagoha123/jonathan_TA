import os

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
from src.rag.retriever import RAGHandler
import base64
from pathlib import Path
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

def get_pdf_data(file_path: str) -> str:
    """Convert PDF to base64 string"""
    try:
        with open(file_path, "rb") as f:
            pdf_data = f.read()
            base64_pdf = base64.b64encode(pdf_data).decode('utf-8')
            return f"data:application/pdf;base64,{base64_pdf}"
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def get_pdf_link(file_path: str) -> str:
    """Generate a download link for PDF"""
    if file_path and file_path.endswith('.pdf'):
        pdf_data = get_pdf_data(file_path)
        if pdf_data:
            filename = os.path.basename(file_path)
            return f'<a href="{pdf_data}" download="{filename}" style="text-decoration:none;color:#2E8BC0;padding:0.5em 1em;border:1px solid #2E8BC0;border-radius:5px;background-color:white;">📥 Download PDF</a>'
    return None

def display_source_documents(message, idx):
    if "source_docs" in message:
        st.divider()
        st.markdown("**Relevant Document:**")
        for doc in message["source_docs"]:
            if doc and doc.get("file_path"):
                pdf_link = get_pdf_link(doc["file_path"])
                if pdf_link:
                    st.markdown(pdf_link, unsafe_allow_html=True)

    
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
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.write(message["content"])
            display_source_documents(message, idx)

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
                    result = rag_handler.generate_response(prompt)
                    st.write(result["response"])
                    
                    if result["context"] and len(result["context"]) > 0:
                        most_relevant_doc = result["context"][0]
                        if most_relevant_doc.get("file_path"):
                            st.divider()
                            st.markdown("**Relevant Document:**")
                            pdf_link = get_pdf_link(most_relevant_doc["file_path"])
                            if pdf_link:
                                st.markdown(pdf_link, unsafe_allow_html=True)
                                                                    
                    # Save message with sources
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": result["response"],
                        "source_docs": [most_relevant_doc] if result["context"] else []
                    })
                    
        except Exception as e:
            st.error(f"Error: {str(e)}")

    # Add sidebar with information
    with st.sidebar:
        st.title("Example Prompts")
        st.markdown("""
        - When is the semester 4 CME assignment deadline?
        - What are the weekly goals for the semester 4 group project?
        - How many credits can I get from an internship?
        - What masters programs can I apply to?
        - What percentage of my semester 4 cme grade is the ethics position statement?
        """)
        
        # Add New Chat button
        if st.button("New Chat"):
            st.session_state.messages = []
            st.rerun()  # Force streamlit to rerun the app

    if "messages" not in st.session_state:
        st.session_state.messages = []

if __name__ == "__main__":
    main()