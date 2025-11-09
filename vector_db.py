import os
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

class VectorDatabase:
    def __init__(self, db_directory="./chroma_db"):
        self.db_directory = db_directory
        self.client = chromadb.PersistentClient(path=db_directory)

        self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-ada-002"
        )
        
        self.collection = self.client.get_or_create_collection(
            name="pdf_documents",
            embedding_function=self.openai_ef
        )
    
    def add_document(self, doc_id, text, metadata=None):
        """Metni vektör veritabanına ekle"""
        chunks = self._chunk_text(text, chunk_size=8000)
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata.update({"chunk_index": i, "parent_doc": doc_id})
            
            self.collection.add(
                documents=[chunk],
                metadatas=[chunk_metadata],
                ids=[chunk_id]
            )
        
        return len(chunks)
    
    def _chunk_text(self, text, chunk_size=8000, overlap=200):
        """Metni belirli boyutlarda parçalara ayır"""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            
            if end < text_length:
                while end > start and text[end] not in ['.', '!', '?', '\n']:
                    end -= 1
                if end == start:
                    end = start + chunk_size
            
            chunks.append(text[start:end])
            start = end - overlap if end < text_length else text_length
        
        return chunks
    
    def search(self, query, n_results=5):
        """Sorguya en uygun belge parçalarını bul"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        return results
    
    def get_document_ids(self):
        """Veritabanındaki tüm belge ID'lerini getir"""
        all_ids = self.collection.get()["ids"]
        # Ana belge ID'lerini ayıkla (chunk ID'lerinden)
        doc_ids = set()
        for id in all_ids:
            if "_chunk_" in id:
                doc_ids.add(id.split("_chunk_")[0])
            else:
                doc_ids.add(id)
        
        return list(doc_ids) 