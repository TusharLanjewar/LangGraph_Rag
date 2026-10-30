import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from core.document_processor import DocumentProcessor
from core.vector_store import VectorStore
from core.rag_graph import RAGGraph


def main():
    
    load_dotenv()
    
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not found in environment variables")
        print("Please set it in a .env file or environment")
        return
    
    print("=" * 60)
    print("RAG Application using LangChain LangGraph")
    print("=" * 60)
    
    doc_processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
    vector_store = VectorStore(persist_directory="./data/chroma_db", collection_name="rag_documents")
    
    print("\n[Step 1] Loading documents...")
    documents_path = "./data/docs"
    
    if not os.path.exists(documents_path):
        print(f"Creating documents directory: {documents_path}")
        os.makedirs(documents_path)
        print(f"Please add your PDF documents to the {documents_path} folder")
        return
    
    try:
        documents = doc_processor.process_pipeline(documents_path, source_type="directory")
        print(f"✓ Loaded and split {len(documents)} document chunks")
    except Exception as e:
        print(f"Error loading documents: {e}")
        print("Make sure you have PDF files in the ./data/docs directory")
        return
    
    print("\n[Step 2] Creating vector store and embeddings...")
    try:
        vector_store.create_vector_store(documents)
        print("✓ Vector store created successfully")
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return
    
    print("\n[Step 3] Initializing LangGraph RAG workflow...")
    rag_graph = RAGGraph(vector_store)
    print("✓ RAG graph initialized")
    
    graph_info = rag_graph.get_graph_info()
    print(f"\nGraph Structure:")
    print(f"  Nodes: {graph_info['nodes']}")
    print(f"  Edges: {graph_info['edges']}")
    
    print("\n" + "=" * 60)
    print("RAG System Ready! Enter your queries (type 'quit' to exit)")
    print("=" * 60)
    
    while True:
        query = input("\nYour question: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not query:
            continue
        
        print("\n[Processing] Retrieving relevant documents and generating answer...")
        
        try:
            result = rag_graph.run(query)
            
            messages = result["messages"]
            ai_message = [msg for msg in messages if hasattr(msg, 'content')][-1]
            
            print(f"\nAnswer: {ai_message.content}")
            print(f"\n[Used {len(result['context'])} document chunks for context]")
            
        except Exception as e:
            print(f"Error processing query: {e}")


if __name__ == "__main__":
    main()
