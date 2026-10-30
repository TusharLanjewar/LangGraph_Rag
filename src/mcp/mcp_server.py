"""FastMCP Server"""

import os
import sys
from typing import Optional

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from fastmcp import FastMCP
from core.document_processor import DocumentProcessor
from core.vector_store import VectorStore
from core.rag_graph import RAGGraph
from langchain_core.messages import AIMessage


mcp = FastMCP("RAG Server")

doc_processor = DocumentProcessor()
vector_store = None
rag_graph = None


@mcp.tool()
def initialize_vector_store(documents_path: str, persist_directory: str = "./data/chroma_db") -> str:
    global vector_store, rag_graph
    
    try:
        print(f"Loading documents from: {documents_path}")
        documents = doc_processor.process_pipeline(documents_path, source_type="directory")
        print(f"Processed {len(documents)} document chunks")
        
        vector_store = VectorStore(persist_directory=persist_directory)
        vector_store.create_vector_store(documents)
        
        rag_graph = RAGGraph(vector_store)
        
        return f"Successfully initialized vector store with {len(documents)} chunks from {documents_path}"
    
    except Exception as e:
        return f"Error initializing vector store: {str(e)}"


@mcp.tool()
def load_existing_vector_store(persist_directory: str = "./data/chroma_db") -> str:
    global vector_store, rag_graph
    
    try:
        vector_store = VectorStore(persist_directory=persist_directory)
        vector_store.load_vector_store()
        
        rag_graph = RAGGraph(vector_store)
        
        return f"Successfully loaded vector store from {persist_directory}"
    
    except Exception as e:
        return f"Error loading vector store: {str(e)}"


@mcp.tool()
def query_rag(query: str) -> str:
    global rag_graph
    
    if rag_graph is None:
        return "Error: RAG system not initialized. Call initialize_vector_store or load_existing_vector_store first."
    
    try:
        result = rag_graph.run(query)
        
        messages = result["messages"]
        ai_message = [msg for msg in messages if isinstance(msg, AIMessage)][-1]
        
        context_info = f"\n\n[Retrieved {len(result['context'])} document chunks]"
        
        return ai_message.content + context_info
    
    except Exception as e:
        return f"Error querying RAG system: {str(e)}"


@mcp.tool()
def search_documents(query: str, k: int = 4) -> str:
    global vector_store
    
    if vector_store is None:
        return "Error: Vector store not initialized. Call initialize_vector_store or load_existing_vector_store first."
    
    try:
        docs = vector_store.similarity_search(query, k=k)
        
        result = f"Retrieved {len(docs)} documents:\n\n"
        for i, doc in enumerate(docs, 1):
            result += f"Document {i}:\n{doc.page_content}\n\n"
        
        return result
    
    except Exception as e:
        return f"Error searching documents: {str(e)}"


@mcp.tool()
def add_documents(file_path: str, file_type: str = "pdf") -> str:
    global vector_store, doc_processor
    
    if vector_store is None:
        return "Error: Vector store not initialized. Call initialize_vector_store or load_existing_vector_store first."
    
    try:
        documents = doc_processor.process_pipeline(file_path, source_type=file_type)
        
        vector_store.add_documents(documents)
        
        return f"Successfully added {len(documents)} chunks from {file_path}"
    
    except Exception as e:
        return f"Error adding documents: {str(e)}"


@mcp.tool()
def get_graph_structure() -> str:
    global rag_graph
    
    if rag_graph is None:
        return "Error: RAG graph not initialized."
    
    try:
        info = rag_graph.get_graph_info()
        return f"Graph Structure:\nNodes: {info['nodes']}\nEdges: {info['edges']}"
    
    except Exception as e:
        return f"Error getting graph info: {str(e)}"


if __name__ == "__main__":
    mcp.run()
