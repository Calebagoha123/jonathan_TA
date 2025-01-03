import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from typing import List, Dict
from src.data.preprocessor import DocumentChunk


class EmbeddingsManager:
    def __init__(self, model_name: str ="all-MiniLM-L6-v2"):
        persist_directory = os.path.abspath("data/chroma_db")
        print(f"Using persistence directory: {persist_directory}")
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)
        
        self.model = SentenceTransformer(model_name)
        self.chroma_client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        collection = self.chroma_client.get_or_create_collection("course_materials")
        print(f"Collection size: {collection.count()}")
        self.collection = collection
         
    def embed_chunks(self, chunks: List[DocumentChunk]):
        texts = [chunk.text for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        self.collection.add(
            documents=texts,
            ids=ids,
            metadatas=metadatas
        )
        
    def query_similar(self, query: str, n_results: int = 3) -> List[Dict]:
        results = self.collection.query(
            query_texts = [query],
            n_results=n_results
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
    
    # Test query
    results = embeddings_manager.query_similar("When is the assignment deadline?")
    print(results['documents'][0])