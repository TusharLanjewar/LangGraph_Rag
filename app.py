import os
import sys
import streamlit as st
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from core.document_processor import DocumentProcessor
from core.vector_store import VectorStore
from core.rag_graph import RAGGraph

st.set_page_config(page_title="Chat", page_icon="💬")
load_dotenv()

if "rag_graph" not in st.session_state:
    st.session_state.rag_graph = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def init_system():
    if st.session_state.rag_graph is not None:
        return True
    
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("Missing GOOGLE_API_KEY in .env file")
        return False
    
    docs_path = "./data/docs"
    if not os.path.exists(docs_path):
        st.error("Create ./data/docs/ folder and add PDF files")
        return False
    
    try:
        with st.spinner("Loading documents..."):
            doc_processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
            documents = doc_processor.process_pipeline(docs_path, source_type="directory")
        
        with st.spinner("Creating vector store..."):
            vector_store = VectorStore(persist_directory="./data/chroma_db", collection_name="rag_documents")
            if os.path.exists("./data/chroma_db"):
                vector_store.load_vector_store()
                vector_store.add_documents(documents)
            else:
                vector_store.create_vector_store(documents)
        
        st.session_state.rag_graph = RAGGraph(vector_store)
        return True
    except Exception as e:
        st.error(f"Error: {e}")
        return False


st.title("� Chat")

if not init_system():
    st.stop()

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.rag_graph.run(prompt)
                messages = result["messages"]
                response = [msg for msg in messages if hasattr(msg, 'content')][-1].content
                st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error: {e}")
