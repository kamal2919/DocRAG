"""Configuration module for agentic rag system"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
import uuid

#Load env variables
load_dotenv()

class Config:
    """Configuration class for RAG system"""

    #LLM keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    OPENAI_API_KEY = os.getenv("OPEN_AI_API_KEY")

    #Model config
    LLM_MODEL = "openai:gpt-4o-mini"

    # Document Processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    
    # Default URLs
    DEFAULT_URLS = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/"
    ]

    @classmethod
    def get_llm(cls):
        """Intialize and return the LLM model"""
        os.environ["OPENAI_API_KEY"] = cls.OPENAI_API_KEY
        return init_chat_model(cls.LLM_MODEL)