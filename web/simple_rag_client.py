"""Client wrapper for simple-rag (vector-based RAG)."""

import sys
from pathlib import Path

# Add simple-rag to path
SIMPLE_RAG_DIR = Path(__file__).parent.parent / "simple-rag"
sys.path.insert(0, str(SIMPLE_RAG_DIR))

from src.vector_store import initialize_vector_store
from src.rag_chain import create_rag_chain
from prompts import SYSTEM_PROMPT


class SimpleRAGClient:
    """Client for querying simple-rag."""

    def __init__(self, model_name: str = "qwen3:1.7b"):
        self.model_name = model_name
        self._chain = None
        self._vector_store = None
        self._current_prompt = None

    def _ensure_initialized(self, system_prompt: str) -> None:
        """Initialize or reinitialize the chain if prompt changed."""
        # Initialize vector store once
        if self._vector_store is None:
            chroma_db_path = str(SIMPLE_RAG_DIR / "chroma_db")
            self._vector_store = initialize_vector_store(
                persist_directory=chroma_db_path
            )

        # Recreate chain if prompt changed
        if self._chain is None or self._current_prompt != system_prompt:
            self._chain = create_rag_chain(
                self._vector_store,
                model_name=self.model_name,
                system_prompt=system_prompt
            )
            self._current_prompt = system_prompt

    def query(self, question: str, system_prompt: str = None) -> dict:
        """
        Query the simple-rag system.

        Args:
            question: The question to ask
            system_prompt: Custom system prompt (uses default if None)

        Returns:
            Dict with 'answer' and 'sources' keys
        """
        prompt = system_prompt if system_prompt else SYSTEM_PROMPT
        self._ensure_initialized(prompt)

        result = self._chain({"query": question})

        # Format sources
        sources = []
        for doc in result.get("source_documents", []):
            source_info = {
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata
            }
            sources.append(source_info)

        return {
            "answer": result.get("result", ""),
            "sources": sources
        }
