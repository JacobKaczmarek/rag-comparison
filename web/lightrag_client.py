"""Client for LightRAG (GraphRAG) via HTTP API."""

import httpx
from typing import Optional, Any

from prompts import SYSTEM_PROMPT


class LightRAGClient:
    """Client for querying LightRAG server."""

    def __init__(self, base_url: str = "http://localhost:9621"):
        self.base_url = base_url
        self._client: Optional[httpx.Client] = None

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=300.0)
        return self._client

    def is_available(self) -> bool:
        """Check if the LightRAG server is running."""
        try:
            response = self._get_client().get(f"{self.base_url}/health", timeout=5.0)
            return response.status_code == 200
        except Exception:
            # Try auth-status endpoint as fallback
            try:
                response = self._get_client().get(f"{self.base_url}/auth-status", timeout=5.0)
                return response.status_code == 200
            except Exception:
                return False

    def query(self, question: str, mode: str = "hybrid", return_context: bool = False, system_prompt: Optional[str] = None) -> dict:
        """
        Query the LightRAG system.

        Args:
            question: The question to ask
            mode: Query mode (hybrid, local, global, naive, mix)
            return_context: If True, also return the raw context sent to LLM
            system_prompt: Custom system prompt (uses default if None)

        Returns:
            Dict with 'answer', 'sources', and optionally 'context' keys
        """
        client = self._get_client()
        result: dict[str, Any] = {"answer": "", "sources": [], "context": None}
        prompt = system_prompt if system_prompt else SYSTEM_PROMPT

        # Use streaming endpoint but collect full response
        try:
            response = client.post(
                f"{self.base_url}/query/stream",
                json={
                    "query": question,
                    "mode": mode,
                    "only_need_context": False,
                    "stream": False,
                    "user_prompt": prompt,
                    "include_references": True,
                    "include_chunk_content": True,
                },
                timeout=300.0
            )
            response.raise_for_status()

            # Parse JSON response
            data = response.json()
            result["answer"] = data.get("response", "")

            # Extract sources with chunk content
            sources = []
            for ref in data.get("references", []):
                # content can be a list of strings or a single string
                content_data = ref.get("content", ref.get("chunk_content", ""))
                file_path = ref.get("file_path", "unknown")

                # Some LightRAG servers return a single reference whose `content`
                # contains multiple chunks (list[str]). Preserve all of them so
                # the UI can show multiple sources.
                if isinstance(content_data, list):
                    contents = [c for c in content_data if isinstance(c, str) and c.strip()]
                elif isinstance(content_data, str) and content_data.strip():
                    contents = [content_data]
                else:
                    contents = []

                for chunk_content in contents:
                    # Truncate for display but keep full text for strict answering.
                    display_content = chunk_content[:300] + "..." if len(chunk_content) > 300 else chunk_content
                    sources.append({
                        "content": display_content,
                        "raw_content": chunk_content,
                        "metadata": {"file": file_path}
                    })

            result["sources"] = sources

            # Optionally fetch the raw context
            if return_context:
                try:
                    ctx_response = client.post(
                        f"{self.base_url}/query",
                        json={
                            "query": question,
                            "mode": mode,
                            "only_need_context": True,
                        },
                        timeout=60.0
                    )
                    ctx_response.raise_for_status()
                    ctx_data = ctx_response.json()
                    result["context"] = ctx_data.get("response", "")
                except Exception:
                    result["context"] = "Failed to fetch context"

            return result

        except httpx.HTTPStatusError as e:
            return {"answer": f"Error: HTTP {e.response.status_code} - {e.response.text}"}
        except httpx.ConnectError:
            return {"answer": "Error: Cannot connect to LightRAG server. Is it running on port 9621?"}
        except Exception as e:
            return {"answer": f"Error: {str(e)}"}

    def close(self):
        """Close the HTTP client."""
        if self._client:
            self._client.close()
            self._client = None
