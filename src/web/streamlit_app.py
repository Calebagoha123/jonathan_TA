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
        welcome_message = """
        👋 Hello! I'm Jonathan, your CSSci course assistant. Ask me anything about your courses, assignments, or deadlines!

        **Important Notes:**
        - For the most up-to-date information, especially regarding deadlines, always refer to Canvas
        - To get the best results, please be as specific as possible in your questions:
          - Include the semester (e.g., "Semester 4")
          - Specify the assignment type (e.g., "individual assignment" or "group project")
          - Mention the specific assignment if applicable
        
        Example: "What are the requirements for the Semester 6 individual reflection essay?"
        """
        st.session_state.messages = [
            {"role": "assistant", "content": welcome_message}
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

def handle_user_input(rag_handler):
    if prompt := st.chat_input("Ask your question here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        try:
            response = st.chat_message("assistant")
            message_placeholder = response.empty()
            full_response = ""
            
            # Stream the response
            for chunk in rag_handler.generate_response(prompt, st.session_state.messages):
                if chunk:
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
            # Add source documents if available
            contexts = rag_handler._get_relevant_context(prompt)
            if contexts and len(contexts) > 0:
                most_relevant_doc = contexts[0]
                if most_relevant_doc.get("file_path"):
                    response.divider()
                    response.markdown("**Relevant Document:**")
                    pdf_link = get_pdf_link(most_relevant_doc["file_path"])
                    if pdf_link:
                        response.markdown(pdf_link, unsafe_allow_html=True)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,
                "source_docs": [most_relevant_doc] if contexts else []
            })
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            print(f"Full error details: {str(e)}")  # For debugging
    
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
    handle_user_input(rag_handler)


    # Add sidebar with information
    with st.sidebar:
        st.title("Example Prompts")
        st.markdown("""
        - What assignments are there for semester 6?
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