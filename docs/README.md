# Jonathan - CSSci RAG Assistant ([Try It!](https://jonathanai.streamlit.app/))

An open-source RAG-powered chatbot that helps students find information from course materials. Built using modern RAG (Retrieval Augmented Generation) techniques, this assistant can answer questions about assignments, course content, and other academic materials found on Canvas.

### Example Prompts
- "When is the semester 4 CME assignment deadline?"
- "What are the weekly goals for the semester 4 group project?"
- "How many credits can i get from an internship?"
- "What masters programs can I apply to?"
- "What percentage of my semester 4 cme grade is the ethics position statement?"


## Features

- 🤖 Natural conversation interface for course-related queries
- 📚 Processes and indexes course materials (PDFs)
- 🎯 Accurate responses based on official course documentation
- 🔍 Smart filtering for semester and assignment-specific questions
- 🔄 New chat functionality for fresh conversations

## Technical Architecture
#### 1. Document Processing (`src/data/`)
- **Preprocessor**: Handles PDF parsing and text extraction
- **Chunking**: Creates semantic document chunks (1000 tokens, 100 overlap to keep context)
- **Metadata**: Tags content with semester and assignment information for smart retrieval
- **Storage**: Saves processed chunks to `data/processed/`

#### 2. Vector Storage (`src/rag/`)
- **ChromaDB**: Persistent vector database
- **Collections**: Organizes embeddings by document type
- **Metadata Filtering**: Smart filtering system for relevant content
- **Query Processing**: Handles semantic similarity search

#### 3. Embeddings & Retrieval Pipeline
- **Model**: all-MiniLM-L6-v2 (Sentence Transformers)
- **LLM**: gpt-4o-mini
- **Batch Processing**: Efficient document embedding
- **Cache**: Persistent storage of embeddings

#### 4. Web Interface (`src/web/`)
- **Streamlit App**: User-friendly chat interface
- **Session Management**: Maintains conversation context
- **Response Generation**: Formats and displays answers

