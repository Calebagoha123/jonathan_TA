import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from typing import List, Dict
from src.data.preprocessor import DocumentChunk

class SentenceTransformerEmbedding:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(input)
        return embeddings.tolist()

class EmbeddingsManager:
    def __init__(self, model_name: str ="all-MiniLM-L6-v2"):
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        persist_directory = os.path.join(root_dir, "data", "chroma_db")
        
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
        print(f"Collection size after adding chunks: {self.collection.count()}")
    def query_similar(self, query: str, n_results: int = 3) -> List[Dict]:
        print(f"Querying collection with: {query}")
        print(f"Current collection size: {self.collection.count()}")
        
        results = self.collection.query(
            query_texts=[query],
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