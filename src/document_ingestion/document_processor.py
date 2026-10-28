"""Document Processing module for loading and splitting documents"""

from typing import List, Union
from pathlib import Path
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyMuPDFLoader,
    TextLoader,
    PyPDFDirectoryLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import uuid

class DocumentProcessor:
    """Handle document loading and procssing"""

    def __init__(self, chunk_size:int = 500, chunk_overlap:int = 50):
        """
        Initialize document processor

        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap betweem chunks
        """

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap
        )

    def load_from_url(self, url:str) -> List[Document]:
        """Load document(s) from a URL"""
        loader = WebBaseLoader(url)
        return loader.load()

    def load_from_pdf_dir(self, directory: Union[str,Path]) -> List[Document]:
        """Load documents from all pdfs iniside a directory"""
        loader = PyMuPDFLoader(str(directory))
        return loader.load()

    def load_from_txt(self, file_path: Union[str, Path]) -> List[Document]:
        """Load document(s) from a text file"""
        loader = TextLoader(str(file_path), encoding='utf-8')
        return loader.load()


    def load_documents(self, sources: List[str]) -> List[Document]:
        """
        Load documents from urls, PDF directories, or TXT files

        Args:
            sources: list of urls, pdf folder paths or TXT file paths

        Returns:
            List of loaded documents
        """

        docs: List[Document] = []
        for src in sources:
            if src.startswith("http://") or src.startswith("https://"):
                docs.extend(self.load_from_url(src))
                continue

            path = Path("data")
            if path.is_dir():  ## PDF Directory
                docs.extend(self.load_from_pdf_dir(path))
            elif path.suffix.lower() == ".txt":
                docs.extend(self.load_from_txt(path))
            else:
                raise ValueError(
                    f"Unsupported source type: {src}."
                    "Use UEL, .txt file of PDF directory"
                )
        return docs

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks

        Args:
            documents: List of documents to split

        Returns:
            List of split documemts
        """
        return self.splitter.split_documents(documents=documents)

    def process_urls(self, urls: List[str]) -> List[Document]:
        """
        Complete pipeline to load and split documents

        Agrs:
            urls: list of URLs to process

        Returns:
            List of processed document chunks
        """
        docs = self.load_documents(urls)
        return self.split_documents(docs)