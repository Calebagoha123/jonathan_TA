import os
from pathlib import Path
from typing import List, Dict
import PyPDF2
from docx import Document
import json
from datetime import datetime
import shutil

class DocumentLoader:
    def __init__(self, raw_dir: str = "data/raw", processed_dir: str = "data/processed", docs_dir: str = "src/web/static/docs"):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.docs_dir = Path(docs_dir)
        self.docs_dir.mkdir(parents=True, exist_ok=True)

    def copy_to_static(self, file_path: Path) -> Path:
        """Copy document to static directory and return new path."""
        relative_path = file_path.relative_to(self.raw_dir)
        new_path = self.docs_dir / relative_path
        new_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, new_path)
        return new_path
      
    def process_pdf(self, file_path: Path) -> str:
        """takes in a file path, extracts the text from the file and returns it as a string"""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
                
        return text
    
    ### NOTE: future implementation for other file types maybe (txt, docx, ppt)
    
    def extract_metadata(self, file_path: Path, static_path: Path) -> Dict:
        """
        get the metadata from the filepath and filename
        return the data as a dictionary
        """
        print(f"\n[DEBUG] Extracting metadata for file: {file_path}")
        parts = file_path.relative_to(self.raw_dir).parts
        print(f"[DEBUG] File parts: {parts}")

        metadata = {
            "course_code": parts[0] if len(parts) > 1 else "unknown",
            "document_type": parts[1] if len(parts) > 2 else "unknown",
            "filename": file_path.name,
            "filter_key": "_".join(parts),
            "processed_date": datetime.now().isoformat(),
            "file_path": str(static_path),
            "original_path": str(file_path)
        }
        print(f"[DEBUG] Generated metadata: {metadata}")
        return metadata
        
    def load_documents(self) -> List[Dict]:
        """Load and process all documents in the raw directory"""
        print("\n[DEBUG] Starting document loading process")
        print(f"[DEBUG] Raw directory: {self.raw_dir}")
    
        processed_docs = []
        
        for file_path in self.raw_dir.rglob("*"):
            print(f"\n[DEBUG] Processing file: {file_path}")
            if not file_path.is_file():
                print(f"[DEBUG] Skipping non-file: {file_path}")
                continue
            
            try:
                static_path = self.copy_to_static(file_path)
                print(f"[DEBUG] Copied to static path: {static_path}")

                if file_path.suffix.lower() == ".pdf":
                    print(f"[DEBUG] Processing PDF file: {file_path}")
                    text = self.process_pdf(file_path)
                    print(f"[DEBUG] Extracted text length: {len(text) if text else 0} characters")

                 ### NOTE: future implementation for other file types maybe (txt, docx, ppt)
                else:
                    print(f"[DEBUG] Skipping non-PDF file: {file_path}")
                    continue
                
                metadata = self.extract_metadata(file_path, static_path)
                print(metadata)
                
                # Save processed document as a json for easy extraction later
                doc_id = f"{metadata['course_code']}_{metadata['document_type']}_{file_path.stem}"
                processed_path = self.processed_dir / f"{doc_id}.json"
                print(f"[DEBUG] Saving processed document to: {processed_path}")
                
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