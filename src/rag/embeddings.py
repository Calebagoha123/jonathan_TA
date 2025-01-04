import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import re
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from typing import List, Dict
from src.data.preprocessor import DocumentChunk
import tempfile
from chromadb.utils import embedding_functions



class SentenceTransformerEmbedding:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(input)
        return embeddings.tolist()

class EmbeddingsManager:
    def __init__(self, model_name: str ="all-MiniLM-L6-v2"):
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        persist_directory = os.path.join(tempfile.gettempdir(), "chroma_db")
        
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)
        
        embedding_function = SentenceTransformerEmbedding(model_name)
        self.chroma_client = chromadb.PersistentClient(
            path=persist_directory
        )
        
        collection = self.chroma_client.get_or_create_collection(
            name="course_materials",
            embedding_function=embedding_function
        )
        self.collection = collection
        
    def filter_chunks(self, query: str) -> List[str]:
        """Filter chunks based on metadata inferred from the query."""
        query_lower = query.lower()
        
        # Debug: Print all unique metadata values
        all_metadata = self.collection.get()['metadatas']
        unique_assignments = set(m.get('assignment') for m in all_metadata if m.get('assignment'))
        print("Available assignments in metadata:", unique_assignments)
        
        # Skip filtering for non-assignment queries
        if any(keyword in query_lower for keyword in ["internship", "masters", "master's"]):
            return None
            
        # Build filter based on semester and assignment
        filter_value = None
        semester = None
        assignment = None

        if "semester" in query_lower:
            semester_match = re.search(r"semester (\d+)", query_lower)
            if semester_match:
                semester = semester_match.group(1)
        
        # Determine assignment type
        if "group" in query_lower:
            assignment_type = "Group_Project_Group_project"
        elif any(kw in query_lower for kw in ["cme", "re", "ssh", "de"]):
            assignment_type = "Individual_Assignments"
            if "cme" in query_lower:
                assignment_type += "_CME"
            elif "re" in query_lower:
                assignment_type += "_RE"
            elif "ssh" in query_lower:
                assignment_type += "_SSH"
            elif "de" in query_lower:
                assignment_type += "_DE"

        # Build filter value
        if semester and assignment_type:
            filter_value = f"Semester_{semester}_{assignment_type}"
        elif semester:
            filter_value = f"Semester_{semester}"
        elif assignment_type:
            filter_value = f".*_{assignment_type}"

        return {"assignment": filter_value} if filter_value else None
    
    def embed_chunks(self, chunks: List[DocumentChunk]):
        if not chunks:
            return
            
        texts = [chunk.text for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        self.collection.add(
            documents=texts,
            ids=ids,
            metadatas=metadatas
        )
        
    def query_similar(self, query: str, n_results: int = 3) -> List[Dict]:
        """Query the collection after filtering based on metadata."""
        where_filters = self.filter_chunks(query)
    
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filters if where_filters else None
        )
        return results
    
if __name__ == "__main__":
    import sys
    import os
    
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


    from src.data.preprocessor import DocumentPreprocessor
    from src.rag.embeddings import EmbeddingsManager
    
    # Test the embeddings pipeline
    preprocessor = DocumentPreprocessor()
    chunks = preprocessor.process_all_documents()
    
    embeddings_manager = EmbeddingsManager()
    embeddings_manager.embed_chunks(chunks)
    
   # Test different query types
    test_queries = [
        #"When is the semester 4 cme assignment deadline?",
        #"Tell me about internship opportunities",
        #"What are the masters programs i can do?",
        "What are the semester 4 group assignment's weekly goals?"
    ]
    
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