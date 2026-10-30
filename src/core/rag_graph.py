from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from .vector_store import VectorStore
import operator


class RAGState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    context: List[str]
    query: str


class RAGGraph:
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
        self.graph = self._build_graph()
    
    def _retrieve_documents(self, state: RAGState) -> RAGState:
        query = state["query"]
        docs = self.vector_store.similarity_search(query, k=4)
        
        context = [doc.page_content for doc in docs]
        
        state["context"] = context
        return state
    
    def _generate_response(self, state: RAGState) -> RAGState:
        query = state["query"]
        context = state["context"]
        
        context_text = "\n\n".join(context)
        prompt = f"""
        You are a helpful assistant. Use the following context to answer the user's question.
        
        Context:
        {context_text}
        
        Question: {query}
        
        Answer:
        """
        
        response = self.llm.invoke(prompt)
        
        state["messages"].append(AIMessage(content=response.content))
        
        return state
    
    def _should_continue(self, state: RAGState) -> str:
        return "generate"
    
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(RAGState)
        
        workflow.add_node("retrieve", self._retrieve_documents)
        workflow.add_node("generate", self._generate_response)
        
        workflow.set_entry_point("retrieve")
        
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        
        return workflow.compile()
    
    def run(self, query: str) -> dict:
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "context": [],
            "query": query
        }
        
        result = self.graph.invoke(initial_state)
        return result
    
    def get_graph_info(self):
        return {
            "nodes": list(self.graph.nodes),
            "edges": list(self.graph.edges)
        }
