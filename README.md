# RAG Application using LangChain, LangGraph & Gemini

A complete Retrieval-Augmented Generation (RAG) application built with LangChain, LangGraph, FastMCP, and Google Gemini. This project demonstrates the fundamental structure of a RAG system for interview explanations.

## What is RAG?

RAG (Retrieval-Augmented Generation) is an AI framework that combines:
- **Retrieval**: Finding relevant documents from a knowledge base
- **Generation**: Using an LLM to generate answers based on retrieved context

This approach allows LLMs to answer questions using up-to-date, domain-specific information without retraining the model.

## Project Structure

```
RAG_LNAGGRAPH/
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── document_processor.py    # Document loading and chunking
│   │   ├── vector_store.py          # Gemini embeddings and vector database
│   │   └── rag_graph.py             # LangGraph workflow orchestration
│   ├── mcp/
│   │   ├── __init__.py
│   │   └── mcp_server.py            # FastMCP server for tool integration
│   └── __init__.py
├── data/
│   ├── docs/                        # Directory for your documents
│   └── chroma_db/                   # Vector database storage (created automatically)
├── config/
│   └── .env.example                 # Environment variables template
├── main.py                          # Main application entry point
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Key Components

### 1. Document Processor (`document_processor.py`)
- **Purpose**: Load and split documents into chunks
- **Key Functions**:
  - `load_pdf()`: Load PDF files
  - `load_directory()`: Load all documents from a folder
  - `split_documents()`: Split documents into smaller chunks for embedding
- **Why Important**: Large documents can't be embedded as single vectors; chunking enables precise retrieval

### 2. Vector Store (`src/core/vector_store.py`)
- **Purpose**: Store and retrieve document embeddings using Gemini
- **Key Functions**:
  - `create_vector_store()`: Create embeddings using Gemini and store in ChromaDB
  - `similarity_search()`: Find relevant documents based on query
- **Why Important**: Enables semantic search - finding documents by meaning, not keywords
- **Model**: Uses `models/embedding-001` for embeddings

### 3. RAG Graph (`src/core/rag_graph.py`)
- **Purpose**: Orchestrate the RAG pipeline using LangGraph
- **Workflow Nodes**:
  - `retrieve`: Get relevant documents from vector store
  - `generate`: Generate answer using retrieved context
- **Why Important**: LangGraph provides a structured, visual way to define complex AI workflows
- **Model**: Uses `gemini-2.5-flash` for response generation

### 4. MCP Server (`src/mcp/mcp_server.py`)
- **Purpose**: Expose RAG functionality via Model Context Protocol
- **Available Tools**:
  - `initialize_vector_store`: Set up the knowledge base
  - `query_rag`: Ask questions and get answers
  - `search_documents`: Search without generation
  - `add_documents`: Add new documents dynamically
- **Why Important**: MCP allows integration with various AI assistants and tools

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your Google API key in a `.env` file:
```
GOOGLE_API_KEY=your_google_api_key_here
```

Get your API key from: https://makersuite.google.com/app/apikey

## Usage

### Where to Add Documents

Add your PDF documents to the `data/docs/` folder:
```
RAG_LNAGGRAPH/
└── data/
    └── docs/
        ├── document1.pdf
        ├── document2.pdf
        └── ...
```

### Option 1: Streamlit Web Interface (Recommended)

1. Add your PDF documents to the `data/docs/` folder
2. Run the Streamlit app:
```bash
streamlit run app.py
```
3. Open your browser to the URL shown (usually http://localhost:8501)
4. Click "Initialize RAG System" in the sidebar
5. Start chatting with your documents!

### Option 2: Command Line Interface

1. Add your PDF documents to the `data/docs/` folder
2. Run the application:
```bash
python main.py
```
3. Enter your questions when prompted

### Option 3: MCP Server

1. Start the MCP server:
```bash
python src/mcp/mcp_server.py
```

2. Connect from an MCP-compatible client and use the available tools

## RAG Pipeline Explained

### Step 1: Document Ingestion
```
Documents → Load → Split into Chunks
```
- Documents are loaded from files
- Split into smaller chunks (1000 characters with 200 overlap)
- Overlap ensures context continuity between chunks

### Step 2: Embedding and Storage
```
Chunks → Embed → Store in Vector Database
```
- Each chunk is converted to a vector embedding using Gemini
- Vectors are stored in ChromaDB for efficient similarity search
- Embeddings capture semantic meaning of the text
- Uses `models/embedding-001` for high-quality embeddings

### Step 3: Retrieval
```
User Query → Embed → Similarity Search → Relevant Chunks
```
- User question is embedded
- Vector database finds most similar chunks
- Top-k results are retrieved as context

### Step 4: Generation
```
Query + Context → Gemini LLM → Answer
```
- Retrieved context is added to the prompt
- Gemini 2.5 Flash generates answer using the provided context
- Answer is grounded in the retrieved documents
- Fast and efficient response generation

## LangGraph Workflow

The RAG graph consists of two main nodes:

```
[User Query]
     ↓
[retrieve] → Get relevant documents from vector store
     ↓
[generate] → Generate answer using context
     ↓
[Answer]
```

This graph-based approach makes the workflow:
- **Visual**: Easy to understand and modify
- **Extensible**: Simple to add new nodes (e.g., re-ranking, filtering)
- **Debuggable**: Each step can be inspected independently

## Interview Talking Points

### What is RAG?
- RAG combines retrieval systems with LLMs
- Allows LLMs to access external, up-to-date information
- Reduces hallucinations by grounding responses in retrieved context

### Why use LangChain?
- Provides standardized interfaces for document processing
- Supports multiple document loaders and vector stores
- Easy to swap components (e.g., change from Chroma to Pinecone)

### Why use LangGraph?
- Orchestrates complex multi-step workflows
- Provides state management across steps
- Enables conditional branching and loops
- Makes workflows visual and debuggable

### Why use Gemini?
- Cost-effective alternative to OpenAI models
- Fast response times with gemini-2.5-flash
- High-quality embeddings with models/embedding-001
- Seamless integration with LangChain
- Generous free tier for development

### What is MCP?
- Model Context Protocol for tool integration
- Allows AI assistants to call external tools
- Standardized protocol for extensibility
- FastMCP provides Python implementation

## Customization

### Change Chunk Size
```python
doc_processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
```

### Use Different Gemini Model
```python
# In rag_graph.py
self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
```

### Change Embedding Model
```python
# In vector_store.py
self.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
```

### Change Vector Store
Modify `vector_store.py` to use Pinecone, Weaviate, or other vector databases

## Requirements

- Python 3.8+
- Google API key (get from https://makersuite.google.com/app/apikey)
- PDF documents in the `data/docs/` folder

## License

MIT License - Feel free to use for learning and interviews
