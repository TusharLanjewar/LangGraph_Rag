from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document


class DocumentProcessor:
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
    
    def load_pdf(self, file_path: str) -> List[Document]:
        loader = PyPDFLoader(file_path)
        return loader.load()
    
    def load_text(self, file_path: str) -> List[Document]:
        loader = TextLoader(file_path)
        return loader.load()
    
    def load_directory(self, directory_path: str, glob_pattern: str = "**/*.pdf") -> List[Document]:
        loader = DirectoryLoader(
            directory_path,
            glob=glob_pattern,
            loader_cls=PyPDFLoader
        )
        return loader.load()
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        return self.text_splitter.split_documents(documents)
    
    def process_pipeline(self, source: str, source_type: str = "directory") -> List[Document]:
        if source_type == "pdf":
            docs = self.load_pdf(source)
        elif source_type == "text":
            docs = self.load_text(source)
        elif source_type == "directory":
            docs = self.load_directory(source)
        else:
            raise ValueError(f"Unknown source type: {source_type}")
        
        return self.split_documents(docs)
