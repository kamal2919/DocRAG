"""Langgraph nodes for RAG workflow + ReAct Agent inside generate_content"""

import uuid
from typing import List, Optional
from src.state.rag_state import RAGState
from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_core.prompts import PromptTemplate
import wikipedia

class RAGNodes:
    """Contains node functions for RAG workflow"""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self._agent = None #lazy-init agent

    def retrieve_docs(self, state:RAGState) -> RAGState:
        """Classic retriever node"""
        docs = self.retriever.invoke(state.question)
        return RAGState(
            question=state.question,
            retrieved_docs=docs
        )

    def _build_tools(self) -> List[Tool]:
        """Build retriever + wikipedia tools"""

        def retriever_tool_fn(query:str) -> str:
            print(f"Tool call came here1")
            docs: List[Document] = self.retrieve_docs(question=query)

            if not docs:
                return "No documents found"
            merged = []
            for i,d in enumerate(docs, start=1):
                meta = d.metadata if hasattr(d,"metadata") else {}
                title = meta.get("title") or meta.get("Source") or f"doc_{i}"
                merged.append(f"[{i}] {title}\n{d.page_content}")

            return "\n\n".join(merged)

        retriever_tool = Tool(
            name = "retriever",
            description="Fetch passages from indexed corpus",
            func=retriever_tool_fn
        )

        wikipedia.set_user_agent(
            "DocRAG/1.0 (contact: ky691185@gmail.com)"
        )

        wiki = WikipediaQueryRun(
            api_wrapper= WikipediaAPIWrapper(top_k_results=3, lang="en")
        )
        def wikipedia_tool_fn(query: str) -> str:
            return wiki.run(query)

        wikipedia_tool = Tool(
            name="wikipedia",
            description="Search Wikipedia for general knowledge",
            func=wikipedia_tool_fn
        )

        return [retriever_tool,wikipedia_tool]

    def _build_agent(self):
        """ReAct agent with tools"""
        tools = self._build_tools()
        system_prompt = """
        You are a helpful ReAct-style RAG agent.

        - Prefer the retriever tool for questions about indexed documents.
        - Use Wikipedia for general knowledge when appropriate.
        - If Wikipedia fails or is unavailable, continue using other available
        information rather than repeatedly calling the failing tool.
        - Never claim to have verified information using a source that failed.
        - Return a clear final answer.
        """
        # system_prompt = PromptTemplate.from_template(
        #     """
        #     You are a helpful RAG agent. Prefer 'retriever' for user-provided docs; use 'wikipedia' for general knowledge.
        #     Return only the final useful answer
        #     """
        # )
        self._agent = create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=system_prompt,
        )

    def generate_answer(self,state: RAGState) -> RAGState:
        """
        Generate answer using ReAct agent with retriever + wikipedia
        """
        if self._agent is None:
            self._build_agent()

        result = self._agent.invoke({"messages": [HumanMessage(content=state.question)]})

        messages = result.get("messages",[])
        answer: Optional[str] = None
        if messages:
            answer_msg = messages[-1]
            answer = getattr(answer_msg, "content", None)

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=answer or "Could not generate answer"
        )