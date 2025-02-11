import json
import re
import os
from pathlib import Path
from typing import List, Dict
import numpy as np
from pydantic import BaseModel

class DocumentChunk(BaseModel):
    chunk_id: str
    text: str
    metadata: Dict
    
class DocumentPreprocessor:
    def __init__(self, processed_dir: str = "data/processed", chunk_size: int = 1000, chunk_overlap: int = 100):
        self.processed_dir = Path(processed_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def clean_text(self, text: str) -> str:
        """Normalize text by removing irrelevant symbols or content."""
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        text = re.sub(r'[^a-zA-Z0-9.,;:!?()\[\]{}\'\"\n\- ]+', '', text)  # Remove unusual characters
        return text.strip()

    def split_by_sections(self, text: str) -> List[str]:
        """Split text by sections using headings and subheadings."""
        section_pattern = re.compile(r'^(?:[A-Z][A-Z\s]+|[0-9]+\.[0-9]+(?:\.[0-9]+)?)$', re.MULTILINE)
        sections = re.split(section_pattern, text)
        sections = [self.clean_text(s).strip() for s in sections if s.strip()]
        return sections
    
    @staticmethod
    def extract_metadata_from_path(file_path: str) -> Dict:
        processed_path = Path(file_path)
                
        # Read the JSON to get original path structure
        with open(processed_path, 'r') as f:
            doc_data = json.load(f)
        
        static_path = doc_data["metadata"]["file_path"]
        
        # Extract components from filename
        filename = processed_path.stem  # Get filename without extension
        parts = filename.split('_')
            
        metadata = {
            "semester": f"{parts[0]}_{parts[1]}",  # Example: Semester_4
            "assignment_type": '_'.join(parts[2:4]),  # Example: Individual_Assignments
            "assignment": '_'.join(parts[4:]),  # Example: CME
            "file_path": str(static_path)
        }
        return metadata

    def chunk_text(self, text: str) -> List[str]:
        """Split the text into overlapping chunks, preserving semantic coherence."""
        print(f"\n[DEBUG] Chunking text of length: {len(text)}")
        words = text.split()
        chunks = []

        if len(words) <= self.chunk_size:
            print("[DEBUG] Text fits in single chunk")
            return [text]

        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = " ".join(words[i: i + self.chunk_size])
            chunks.append(chunk)
            
        print(f"[DEBUG] Created {len(chunks)} chunks")
        return chunks

    def process_document(self, doc_path: Path) -> List[DocumentChunk]:
        """Process a document into enriched chunks."""
        print(f"\n[DEBUG] Processing document: {doc_path}")
        with open(doc_path, 'r', encoding="utf-8") as f:
            doc_data = json.load(f)

        print("[DEBUG] Loaded JSON data")
        raw_text = self.clean_text(doc_data.get("text", ""))
        sections = self.split_by_sections(raw_text)
        metadata = self.extract_metadata_from_path(doc_path)
        print(f"[DEBUG] Extracted metadata: {metadata}")

        doc_chunks = []
        print(f"[DEBUG] Processing {len(sections)} sections")

        for section_index, section in enumerate(sections):
            print(f"[DEBUG] Processing section {section_index}")
            section_chunks = self.chunk_text(section)
            for chunk_index, chunk_text in enumerate(section_chunks):
                chunk_id = f"{metadata['semester']}_{metadata['assignment_type']}_{metadata['assignment']}_section_{section_index}_chunk_{chunk_index}"
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "section_index": section_index,
                    "chunk_index": chunk_index,
                    "total_chunks_in_section": len(section_chunks),
                    "section_summary": section[:150],
                })
                doc_chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        metadata=chunk_metadata
                    )
                )
        print(f"[DEBUG] Created {len(doc_chunks)} chunks for document")
        return doc_chunks

    def process_all_documents(self) -> List[DocumentChunk]:
        """Process all documents in the processed directory."""
        print("\n[DEBUG] Starting processing of all documents")
        print(f"[DEBUG] Processing directory: {self.processed_dir}")
        all_chunks = []

        for doc_path in self.processed_dir.glob("*.json"):
            try:
                print(f"\n[DEBUG] Processing file: {doc_path}")
                doc_chunks = self.process_document(doc_path)
                all_chunks.extend(doc_chunks)
                print(f"[DEBUG] Total chunks so far: {len(all_chunks)}")

            except Exception as e:
                print(f"[ERROR] Error processing {doc_path}: {str(e)}")
                
        print(f"\n[DEBUG] Finished processing all documents")
        print(f"[DEBUG] Total chunks created: {len(all_chunks)}")
        return all_chunks
    
if __name__ == "__main__":
    preprocessor = DocumentPreprocessor()
    chunks = preprocessor.process_all_documents()
    print(f"Created {len(chunks)} total chunks from all documents")