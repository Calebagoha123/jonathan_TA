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
    if "pdf_visibility" not in st.session_state:
        st.session_state.pdf_visibility = {}

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
    """Generate an embedded PDF viewer link"""
    if file_path and file_path.endswith('.pdf'):
        pdf_data = get_pdf_data(file_path)
        if pdf_data:
            return f'''
                <iframe 
                    src="{pdf_data}" 
                    width="100%" 
                    height="800px"
                    style="border: none;"
                    type="application/pdf"
                    title="PDF Viewer">
                    <object
                        data="{pdf_data}"
                        type="application/pdf"
                        width="100%"
                        height="800px">
                        <p>Your browser does not support PDFs. Please download to view.</p>
                    </object>
                </iframe>
            '''
    return None

def display_source_documents(message, idx):
    if "source_docs" in message:
        st.divider()
        st.markdown("**Relevant Document:**")
        for doc_idx, doc in enumerate(message["source_docs"]):
            if doc and doc.get("file_path"):
                message_id = f"pdf_{idx}_{doc_idx}"
                
                # Get current visibility state
                is_visible = st.session_state.pdf_visibility.get(message_id, False)
                
                # Create toggle button with icon
                if st.button("👁️ View PDF" if not is_visible else "👁️ Hide PDF", 
                           key=message_id):
                    st.session_state.pdf_visibility[message_id] = not is_visible
                    st.rerun()
                
                # Show PDF if visible
                if st.session_state.pdf_visibility.get(message_id, False):
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
                            
                            # Create unique message ID for the new response
                            message_id = f"pdf_{len(st.session_state.messages)}_0"
                            
                            # Initialize visibility state for new message
                            if message_id not in st.session_state.pdf_visibility:
                                st.session_state.pdf_visibility[message_id] = False
                            
                            # Add toggle button
                            if st.button("👁️ View PDF" if not st.session_state.pdf_visibility[message_id] else "👁️ Hide PDF", 
                                    key=message_id):
                                st.session_state.pdf_visibility[message_id] = not st.session_state.pdf_visibility[message_id]
                                st.rerun()
                            
                            # Show PDF if visible
                            if st.session_state.pdf_visibility[message_id]:
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