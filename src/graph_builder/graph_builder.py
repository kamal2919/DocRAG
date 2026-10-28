"""Graph builder for langgraph workflow"""

from langgraph.graph import StateGraph, END
from src.state.rag_state import RAGState
# from src.nodes.nodes import RAGNodes
from src.nodes.reactnode import RAGNodes
import uuid

class GraphBuilder:
    """Builds and manage the LangGraph workflow"""

    def __init__(self,retriever,llm):
        """
        Intialize graph builder

        Args:
            retriever: Document retirver instance
            llm: Language Model Instance
        """
        self.nodes = RAGNodes(retriever,llm)
        self.graph = None

    def build(self):
        """
        Build the RAG workflow graph

        Returns:
            Compiled graph instance
        """
        ## Create state graph
        builder = StateGraph(RAGState)

        # Add nodes
        builder.add_node("retriever", self.nodes.retrieve_docs)
        builder.add_node("responder",self.nodes.generate_answer)

        # Set answer point
        builder.set_entry_point("retriever")

        # Add edges
        builder.add_edge("retriever","responder")
        builder.add_edge("responder",END)

        # compile graph
        self.graph = builder.compile()
        return self.graph

    def run(self,question:str) -> dict:
        """
        Run the RAG workflow

        Args:
            question: User question
        
        Returns:
            Final state with answer
        """
        if self.graph is None:
            self.build()

        initial_state = RAGState(question=question)
        return self.graph.invoke(initial_state)