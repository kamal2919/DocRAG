"""Langgraph nodes for RAG Workflow"""

from src.state.rag_state import RAGState
import uuid

class RAGNodes:
    """Contains node functions for RAG Workflow"""

    def __init__(self,retriever, llm):
        """
        Initialize RAG Nodes

        Args:
            retriever: Document retriever instance
            llm: languages model instance
        """
        self.retriever = retriever
        self.llm = llm

    def retrieve_docs(self,state:RAGState) -> RAGState:
        """
        Retrieve relevant documents nod

        Args:
            state: Current RAG state
        
        Returns:
            Updated RAG state with retrieved docs
        """

        docs = self.retriever.invoke(state.question)
        return RAGState(
            question=state.question,
            retrieved_docs=docs
        )

    def generate_answer(self, state:RAGState) -> RAGState:
        """
        Generate answer from retrieved documents node

        Args:
            state: Current RAG state with retrieved documents

        Returns:
            Updated RAG state with generate answer
        """

        # Combine retrieved documents into context
        context = "\n\n".join([doc.page_content for doc in state.retrieved_docs])

        # Create prompt
        prompt = f"""
        Answer the question based on the context.

        Context:
        {context}

        Question:
        {state.question}
        """

        # Generate response
        response = self.llm.invoke(prompt)

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=response.content
        )