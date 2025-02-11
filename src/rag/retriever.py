import os
import sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import List, Dict
from openai import OpenAI
from src.rag.embeddings import EmbeddingsManager
from dotenv import load_dotenv

load_dotenv()

class RAGHandler:
    def __init__(self):
        self.embeddings_manager = EmbeddingsManager()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def _create_prompt(self, query: str, contexts: List[Dict], conversation_history: List[Dict]) -> str:
        semester = "semester 6" if any("Semester_6" in ctx["metadata"]["semester"] 
                                 for ctx in contexts) else ""
        """Create a prompt to ask the the llm using the context retrieved"""
        
        # Format conversation history
        formatted_history = "\n".join([
            f"{'Student' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
            for msg in conversation_history[:-1]  # Exclude current query
        ])
        context_texts = [ctx["text"] for ctx in contexts]
        prompt = f"""As Jonathan, the CSSci course assistant, use the following {semester} course material to answer the student's question. For capstone semester questions, consider the interconnected 
        nature of all deliverables. If the answer cannot be found in the context, say so clearly.
    
        Previous conversation:
        {formatted_history}
        
        Relevant course material:
        {' '.join(context_texts)}
        
        Student question: {query}
        Assistant response:"""
        
        return prompt
    
    def _get_relevant_context(self, query: str, n_results: int = 3) -> List[Dict]:
        """Get the relevant context from the vector store"""
        results = self.embeddings_manager.query_similar(query, n_results=n_results)
        documents = []
        print("\nDebug - Raw results from ChromaDB:")
        print(f"Metadatas: {results['metadatas']}")
        
        for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
            documents.append({
                "text": doc,
                "metadata": metadata,
                "file_path": metadata.get('file_path') if metadata else None
            }) 
            print(f"Created document with file_path: {metadata.get('file_path')}")
        return documents
    
    def generate_response(self, query: str, conversation_history: List[Dict]) -> str:
        """Generate the llm's response using RAG"""
        # get the relevant context
        context = self._get_relevant_context(query)
        
        # create the prompt using the context
        prompt = self._create_prompt(query, context, conversation_history)
        
        # generate response using llm
        # currently using OpenAI's GPT 4o-mini turbo (future development: locally ran small scale open source llm**)
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You Jonathan, are a helpful Computational Social Science (CSSci) course assistant that helps students understand course materials."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7, # might have to fine tune this based on testing
            max_tokens = 500, # this as well (short answers atm)
            stream=True
        )
        
        for chunk in response:
            delta = chunk.choices[0].delta  # access the delta attribute directly
            content = getattr(delta, "content", "")
            yield content
    
    def chat(self, query: str) -> str:
        """Simple chat interface"""
        try:
            result = self.generate_response(query)
            return ''.join(result)
        except Exception as e:
            return f"Error generating response: {str(e)}"
        
if __name__ == "__main__":
    # Test the RAG system
    rag = RAGHandler()
    
    # Test questions
    test_questions = [
"What are the details of the capstone final product?", 
"What are the weekly goals for the semester 4 group project?",
"When is the semester 4 CME assignment deadline?"
    ]
    
    print("Testing RAG system with sample questions:\n")
    for question in test_questions:
        print(f"Q: {question}")
        print(f"A: {rag.chat(question)}\n")