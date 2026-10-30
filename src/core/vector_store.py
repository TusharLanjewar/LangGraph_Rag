from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List, Optional
import os


class VectorStore:
    
    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "rag_documents"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
        self.vector_store: Optional[Chroma] = None
    
    def create_vector_store(self, documents: List[Document]) -> Chroma:
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name
        )
        return self.vector_store
    
    def load_vector_store(self) -> Chroma:
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )
        return self.vector_store
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        if self.vector_store is None:
            self.load_vector_store()
        
        return self.vector_store.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        if self.vector_store is None:
            self.load_vector_store()
        
        return self.vector_store.similarity_search_with_score(query, k=k)
    
    def add_documents(self, documents: List[Document]):
        if self.vector_store is None:
            self.load_vector_store()
        
        self.vector_store.add_documents(documents)
