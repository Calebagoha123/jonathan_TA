import os
from pathlib import Path
from typing import List, Dict
import PyPDF2
from docx import Document
import json
from datetime import datetime

class DocumentLoader:
    def __init__(self, raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def process_pdf(self, file_path: Path) -> str:
        """takes in a file path, extracts the text from the file and returns it as a string"""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
                
        return text
    
    ### NOTE: future implementation for other file types maybe (txt, docx, ppt)
    
    def extract_metadata(self, file_path: Path) -> Dict:
        """
        get the metadata from the filepath and filename
        return the data as a dictionary
        """
        # this function assumes that the file is stored like this: raw/COURSE_CODE/DOCUMENT_TYPE/filename
        # so that we can easily keep track of which document belongs to which course, etc.
        parts = file_path.relative_to(self.raw_dir).parts
        
        return {
            "course_code": parts[0] if len(parts) > 1 else "unknown",
            "document_type": parts[1] if len(parts) > 2 else "unknown",
            "filename": file_path.name,
            "processed_date": datetime.now().isoformat()
        }
        
    def load_documents(self) -> List[Dict]:
        """Load and process all documents in the raw directory"""
        processed_docs = []
        
        for file_path in self.raw_dir.rglob("*"):
            if not file_path.is_file():
                continue
            
            try:
                if file_path.suffix.lower() == ".pdf":
                    text = self.process_pdf(file_path)
                 ### NOTE: future implementation for other file types maybe (txt, docx, ppt)
                else:
                    continue
                
                metadata = self.extract_metadata(file_path)
                
                # Save processed document as a json for easy extraction later
                doc_id = f"{metadata['course_code']}_{metadata['document_type']}_{file_path.stem}"
                processed_path = self.processed_dir / f"{doc_id}.json"
                
                doc_data = {
                    "id": doc_id,
                    "text": text,
                    "metadata": metadata
                }
                
                with open(processed_path, 'w', encoding='utf-8') as f:
                    json.dump(doc_data, f, ensure_ascii=False, indent=2)
                
                processed_docs.append(doc_data)
                print(f"Processed: {file_path.name}")
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                
        return processed_docs
            
            
if __name__ == "__main__":
    loader = DocumentLoader()
    processed_docs = loader.load_documents()
    print(f"Successfully processed {len(processed_docs)} documents")