import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import re
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from typing import List, Dict
from src.data.preprocessor import DocumentChunk

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class SentenceTransformerEmbedding:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(input)
        return embeddings.tolist()

class EmbeddingsManager:
    def __init__(self, model_name: str ="all-MiniLM-L6-v2"):
        print(f"\n[DEBUG] Initializing EmbeddingsManager with model: {model_name}")
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        persist_directory = os.path.join(root_dir, "data", "chroma_db")
        
        print(f"[DEBUG] Using persist directory: {persist_directory}")
        if not os.path.exists(persist_directory):
            print("[DEBUG] Creating persist directory")
            os.makedirs(persist_directory)
        
        print("[DEBUG] Initializing embedding function")
        embedding_function = SentenceTransformerEmbedding(model_name)
        
        print("[DEBUG] Creating ChromaDB client")
        self.chroma_client = chromadb.PersistentClient(
            path=persist_directory
        )
        
        print("[DEBUG] Getting or creating collection")
        collection = self.chroma_client.get_or_create_collection(
            name="course_materials",
            embedding_function=embedding_function
        )
        self.collection = collection
        
    def filter_chunks(self, query: str) -> List[str]:
        """Filter chunks based on metadata inferred from the query."""
        print(f"\n[DEBUG] Filtering chunks for query: {query}")
        query_lower = query.lower()
        
        # Skip filtering for non-assignment queries
        if any(keyword in query_lower for keyword in ["internship", "masters", "master's"]):
            print("[DEBUG] Skipping filters for general query")
            return None
            
        # Build filter based on semester and assignment
        print("[DEBUG] Building metadata filters")
        filters = {}
        
        if "semester 6" in query_lower or "capstone" in query_lower:
            print("[DEBUG] Detected Semester 6 filter")
            filters['semester'] = "Semester_6"
            return filters

        if "semester" in query_lower:
            semester_match = re.search(r"semester (\d+)", query_lower)
            if semester_match:
                semester_num = semester_match.group(1)
                filters['semester'] = f"Semester_{semester_num}"
                print(f"[DEBUG] Detected semester: {filters['semester']}")

        
        # Determine assignment type
        print("[DEBUG] Determining assignment type")
        if "group" and "individual" in query_lower:
            filters['assignment_type'] = "Group_Project"
            filters['assignment'] = "individual_contribution"
        elif "group" in query_lower:
            filters['assignment_type'] = "Group_Project"
            filters['assignment'] = "Group_project"
        elif any(kw in query_lower for kw in ["cme", "re", "ssh", "de"]):
            filters['assignment_type'] = "Individual_Assignments"
            if "cme" in query_lower:
                filters['assignment'] = "CME"
            elif "re" in query_lower:
                filters['assignment'] = "RE"
            elif "ssh" in query_lower:
                filters['assignment'] = "SSH"
            elif "de" in query_lower:
                filters['assignment'] = "DE"
        else:
            return None
            
        print(f"[DEBUG] Final filter value: {filters}")
        return {"filter_key": "_".join(filters.values())} if filters else None
    
    def embed_chunks(self, chunks: List[DocumentChunk]):
        print(f"\n[DEBUG] Embedding {len(chunks) if chunks else 0} chunks")
        if not chunks:
            print("[DEBUG] No chunks to embed")
            return
            
        texts = [chunk.text for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        print(f"[DEBUG] Preparing to add {len(texts)} documents to collection")
  
        self.collection.add(
            documents=texts,
            ids=ids,
            metadatas=metadatas
        )
        print("[DEBUG] Successfully added chunks to collection")
        
    def query_similar(self, query: str, n_results: int = 3) -> List[Dict]:
        """Query the collection after filtering based on metadata."""
        print(f"\n[DEBUG] Processing query: {query}")
        where_filters = self.filter_chunks(query)
        print(f"[DEBUG] Using filters: {where_filters}")

        print(f"[DEBUG] Querying collection for {n_results} results")
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filters if where_filters else None
        )
        print(f"[DEBUG] Found {len(results['documents'])} matching documents")
        return results
    
if __name__ == "__main__":
    print("\n[DEBUG] Starting embeddings pipeline test")

    import sys
    import os
    
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


    from src.data.preprocessor import DocumentPreprocessor
    from src.rag.embeddings import EmbeddingsManager
    
    # Test the embeddings pipeline
    preprocessor = DocumentPreprocessor()
    print("[DEBUG] Created DocumentPreprocessor")

    chunks = preprocessor.process_all_documents()
    print(f"[DEBUG] Processed {len(chunks)} document chunks")

    embeddings_manager = EmbeddingsManager()
    print("[DEBUG] Created EmbeddingsManager")

    print("[DEBUG] Embedding chunks")
    embeddings_manager.embed_chunks(chunks)
    
   # Test different query types
    test_queries = [
        "What are the details of the capstone final product?", 
        "Tell me about internship opportunities",
        "What are the masters programs i can do?",
        "What are the semester 4 group assignment's weekly goals?"
    ]
    
    print("\n[DEBUG] Testing queries")
    for query in test_queries:
        print(f"\nTesting query: {query}")
        filters = embeddings_manager.filter_chunks(query)
        print(f"Generated filters: {filters}")
        
        results = embeddings_manager.query_similar(query) 
        print(f"Number of results: {len(results['documents'])}")
        if results['documents']:
            print(f"First result: {results['documents'][0]}")
        else:
            print("No results found")