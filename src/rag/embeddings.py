import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import re
from openai import OpenAI
import chromadb
from chromadb.config import Settings
from typing import List, Dict
from src.data.preprocessor import DocumentChunk
from dotenv import load_dotenv

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class OpenAIEmbedding:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = []
        for text in input:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            embeddings.append(response.data[0].embedding)
        return embeddings

class EmbeddingsManager:
    def __init__(self):
        print(f"\n[DEBUG] Initializing EmbeddingsManager with OpenAI embeddings")
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.persist_directory = os.path.join(root_dir, "data", "chroma_db")
        
        print(f"[DEBUG] Using persist directory: {self.persist_directory}")
        if not os.path.exists(self.persist_directory):
            print("[DEBUG] Creating persist directory")
            os.makedirs(self.persist_directory)
        
        print("[DEBUG] Initializing embedding function")
        self.embedding_function = OpenAIEmbedding()
        
        print("[DEBUG] Creating ChromaDB client")
        self.chroma_client = chromadb.PersistentClient(
            path=self.persist_directory
        )
        
        print("[DEBUG] Getting or creating collection")
        self.collection = self.chroma_client.get_or_create_collection(
            name="course_materials",
            embedding_function=self.embedding_function
        )
        
    def reset_collection(self):
        """Reset the collection by deleting and recreating it."""
        print("[DEBUG] Resetting collection")
        try:
            self.chroma_client.delete_collection("course_materials")
            print("[DEBUG] Deleted existing collection")
        except Exception as e:
            print(f"[DEBUG] No existing collection to delete: {e}")
        
        print("[DEBUG] Creating new collection")
        self.collection = self.chroma_client.create_collection(
            name="course_materials",
            embedding_function=self.embedding_function
        )
        
    def filter_chunks(self, query: str) -> List[str]:
        """Filter chunks based on metadata inferred from the query."""
        print(f"\n[DEBUG] Filtering chunks for query: {query}")
        query_lower = query.lower()
        
        # Skip filtering for non-assignment queries
        if any(keyword in query_lower for keyword in ["internship", "masters", "master's", "career"]):
            print("[DEBUG] Skipping filters for general query")
            return None
            
        # Build filter based on semester and assignment
        print("[DEBUG] Building metadata filters")
        filters = {}
        
        # Assignment type filtering (do this first as it's more specific)
        print("[DEBUG] Determining assignment type")
        assignment_keywords = {
            "cme": ["cme", "continuous monitoring", "monitoring evaluation"],
            "re": ["re", "requirements engineering", "requirements elicitation"],
            "ssh": ["ssh", "system security", "security hardening"],
            "de": ["de", "data engineering", "data pipeline"]
        }
        
        # Special case for capstone/final product
        if "capstone" in query_lower or "final product" in query_lower:
            filters['assignment_type'] = "Group_Project"
            filters['assignment'] = "final_product"
            print("[DEBUG] Detected capstone/final product")
        else:
            for assignment_type, keywords in assignment_keywords.items():
                if any(kw in query_lower for kw in keywords):
                    filters['assignment_type'] = "Individual_Assignments"
                    filters['assignment'] = assignment_type.upper()
                    print(f"[DEBUG] Detected assignment type: {assignment_type.upper()}")
                    break
        
        # Group project filtering
        if "group" in query_lower:
            if "individual" in query_lower or "contribution" in query_lower:
                filters['assignment_type'] = "Group_Project"
                filters['assignment'] = "individual_contribution"
            else:
                filters['assignment_type'] = "Group_Project"
                filters['assignment'] = "Group_project"
            print(f"[DEBUG] Detected group project type: {filters['assignment']}")
        
        # Semester filtering
        if "semester 6" in query_lower or "capstone" in query_lower:
            print("[DEBUG] Detected Semester 6 filter")
            filters['semester'] = "Semester_6"
        elif "semester" in query_lower:
            semester_match = re.search(r"semester (\d+)", query_lower)
            if semester_match:
                semester_num = semester_match.group(1)
                filters['semester'] = f"Semester_{semester_num}"
                print(f"[DEBUG] Detected semester: {filters['semester']}")

        # If we have filters, construct the filter conditions
        if filters:
            print(f"[DEBUG] Final filter value: {filters}")
            
            # Build the filter conditions based on the metadata structure
            where_conditions = []
            
            # Add exact match for the complete filter key
            filter_key = "_".join(filters.values())
            where_conditions.append({"filter_key": filter_key})
            
            # Add individual field matches
            if 'semester' in filters:
                where_conditions.append({"semester": filters['semester']})
            if 'assignment_type' in filters:
                where_conditions.append({"assignment_type": filters['assignment_type']})
            if 'assignment' in filters:
                where_conditions.append({"assignment": filters['assignment']})
            
            # Return the combined filter conditions
            return {"$or": where_conditions}
        
        print("[DEBUG] No filters applied")
        return None
    
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

        # Increase initial results when filtering to ensure we get enough relevant matches
        actual_n_results = n_results * 2 if where_filters else n_results
        
        print(f"[DEBUG] Querying collection for {actual_n_results} results")
        results = self.collection.query(
            query_texts=[query],
            n_results=actual_n_results,
            where=where_filters if where_filters else None
        )
        
        # If we got too many results, trim them down
        if len(results['documents'][0]) > n_results:
            for key in results.keys():
                if isinstance(results[key], list):
                    results[key] = [results[key][0][:n_results]]
        
        print(f"[DEBUG] Found {len(results['documents'][0])} matching documents")
        return results
    
    def log_query_performance(self, query: str, results: Dict, filters_used: Dict):
        """Log query performance for monitoring and improvement."""
        print(f"\n[PERFORMANCE] Query: {query}")
        print(f"[PERFORMANCE] Filters used: {filters_used}")
        print(f"[PERFORMANCE] Results found: {len(results['documents'][0])}")
        print(f"[PERFORMANCE] Top match scores: {results['distances'][0]}")
    
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
        print(f"Number of results: {len(results['documents'][0])}")
        if results['documents'][0]:
            print(f"First result: {results['documents'][0][0]}")
        else:
            print("No results found")