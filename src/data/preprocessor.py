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
        path_parts = Path(file_path).parts
        metadata = {
            "semester": path_parts[-3],  # Example: Semester_4
            "assignment_type": path_parts[-2],  # Example: Individual_Assignments
            "assignment": os.path.splitext(path_parts[-1])[0],  # Example: CME
        }
        return metadata

    def chunk_text(self, text: str) -> List[str]:
        """Split the text into overlapping chunks, preserving semantic coherence."""
        words = text.split()
        chunks = []

        if len(words) <= self.chunk_size:
            return [text]

        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = " ".join(words[i: i + self.chunk_size])
            chunks.append(chunk)

        return chunks

    def process_document(self, doc_path: Path) -> List[DocumentChunk]:
        """Process a document into enriched chunks."""
        with open(doc_path, 'r', encoding="utf-8") as f:
            doc_data = json.load(f)

        raw_text = self.clean_text(doc_data.get("text", ""))
        sections = self.split_by_sections(raw_text)
        metadata = self.extract_metadata_from_path(doc_path)
        doc_chunks = []

        for section_index, section in enumerate(sections):
            section_chunks = self.chunk_text(section)
            for chunk_index, chunk_text in enumerate(section_chunks):
                chunk_id = f"{metadata['assignment']}_section_{section_index}_chunk_{chunk_index}"
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
        return doc_chunks

    def process_all_documents(self) -> List[DocumentChunk]:
        """Process all documents in the processed directory."""
        all_chunks = []

        for doc_path in self.processed_dir.glob("*.json"):
            try:
                doc_chunks = self.process_document(doc_path)
                all_chunks.extend(doc_chunks)
                print(f"Processed {doc_path.name} into {len(doc_chunks)} chunks.")
            except Exception as e:
                print(f"Error processing {doc_path}: {str(e)}")

        return all_chunks
    
if __name__ == "__main__":
    preprocessor = DocumentPreprocessor()
    chunks = preprocessor.process_all_documents()
    print(f"Created {len(chunks)} total chunks from all documents")