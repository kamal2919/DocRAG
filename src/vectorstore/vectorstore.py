"""Vector store module for document embedding and retrieval"""

from typing import List
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import uuid

class VectoreStore:
    """Manages vector store operations"""

    def __init__(self):
        """Intialize vector store with OpenAI embeddings"""
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None
        self.retriever = None

    def create_vectorestore(self, documents: List[Document], k:int=4):
        """
        Create vector store from documents

        Args:
            documents: List of documents to embedd
            k: Number of documents to retrieve
        """
        self.vectorstore = FAISS.from_documents(documents=documents, embedding=self.embeddings)
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs = {"k":k}
        )

    def get_retriever(self):
        """
        Get the vector store retriever instance

        Returns:
            Vector store retriever instance
        """
        if self.retriever is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        return self.retriever

    def retrieve(self, query:str) -> list[Document]:
        """
        Retrieve relevant documents for a query

        Args:
            query: Search Query
        
        Return:
            List of relevenat documents
        """
        if self.retriever is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        return self.retriever.invoke(query)