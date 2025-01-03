import json
from pathlib import Path
from typing import List, Dict
import numpy as np
from pydantic import BaseModel

class DocumentChunk(BaseModel):
    chunk_id: str
    text: str
    metadata: Dict
    
class DocumentPreprocessor:
    def __init__(self, processed_dir: str = "data/processed", chunk_size: int = 500, chunk_overlap: int = 50):
        self.processed_dir = Path(processed_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def chunk_text(self, text: str) -> List[str]:
        """Split the text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        if len(words) <= self.chunk_size:
            return [text]
        
        for i in range (0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = " ".join(words[i : i + self.chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    def process_document(self, doc_path: Path) -> List[DocumentChunk]:
        """process a full document into chunks"""
        with open(doc_path, 'r', encoding="utf-8") as f:
            doc_data = json.load(f)
            
        chunks = self.chunk_text(doc_data['text'])
        doc_chunks = []
        
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{doc_data['id']}_chunk_{i}"
            
            # update metadata with chunk information
            chunk_metadata = doc_data['metadata'].copy()
            chunk_metadata.update({
                "chunk_id": chunk_id,
                "chunk_index": i,
                "total_chunks": len(chunks)
            })
            
            doc_chunks.append(
                DocumentChunk(
                    chunk_id = chunk_id,
                    text = chunk_text,
                    metadata = chunk_metadata
                )
            )
            
        return doc_chunks
        
        
    def process_all_documents(self) -> List[DocumentChunk]:
        """Process all the documents in the 'processed' directory"""
        all_chunks = []
        
        for doc_path in self.processed_dir.glob("*.json"):
            try:
                doc_chunks = self.process_document(doc_path)
                all_chunks.extend(doc_chunks)
                print(f"Processed {doc_path.name} into {len(doc_chunks)} chunks")
            except Exception as e:
                print(f"Error processing {doc_path}: {str(e)}")
                
        return all_chunks
    
if __name__ == "__main__":
    preprocessor = DocumentPreprocessor()
    chunks = preprocessor.process_all_documents()
    print(f"Created {len(chunks)} total chunks from all documents")