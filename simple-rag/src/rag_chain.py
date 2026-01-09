"""RAG chain setup using LangChain and Ollama."""

from typing import Dict, Any, List
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma


class RAGChain:
    """Wrapper for RAG chain using LCEL."""

    def __init__(self, chain, retriever):
        self.chain = chain
        self.retriever = retriever

    def __call__(self, inputs: Dict[str, str]) -> Dict[str, Any]:
        """Execute the chain and return results with source documents."""
        query = inputs.get("query", "")

        # Get source documents
        source_docs = self.retriever.invoke(query)

        # Run the chain
        result = self.chain.invoke({"question": query})

        return {
            "result": result,
            "source_documents": source_docs
        }


def create_rag_chain(vector_store: Chroma, model_name: str = "qwen3:1.7b", system_prompt: str = None) -> RAGChain:
    """
    Create a RAG chain with Ollama LLM and vector store retriever.

    Args:
        vector_store: ChromaDB vector store instance
        model_name: Ollama model to use (default: qwen3:8b)

    Returns:
        Configured RAG chain

    Raises:
        Exception: If Ollama connection fails
    """
    print(f"Initializing Ollama with model: {model_name}")

    # Initialize Ollama LLM
    try:
        llm = Ollama(
            model=model_name,
            temperature=0.7,
            num_ctx=4096,
        )
        # Test connection
        llm.invoke("test")
        print("Connected to Ollama successfully")
    except Exception as e:
        raise Exception(
            f"Failed to connect to Ollama: {e}\n\n"
            "Please ensure:\n"
            "1. Ollama is running: ollama serve\n"
            f"2. Model is installed: ollama pull {model_name}"
        )

    # Create retriever from vector store
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}  # Retrieve top 4 most relevant chunks
    )

    # Create custom prompt template
    default_prompt = """Answer the question based only on the provided context.
If you don't know the answer based on the context, say "I don't know" - don't make up information.
Be concise and direct in your response."""

    prompt_instruction = system_prompt if system_prompt else default_prompt

    template = f"""{prompt_instruction}

Context:
{{context}}

Question: {{question}}

Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    # Format retrieved documents
    def format_docs(docs):
        formatted = []
        for doc in docs:
            if hasattr(doc, 'page_content'):
                formatted.append(str(doc.page_content))
            elif isinstance(doc, dict) and 'page_content' in doc:
                formatted.append(str(doc['page_content']))
            else:
                formatted.append(str(doc))
        return "\n\n".join(formatted)

    # Create RAG chain using LCEL
    rag_chain = (
        RunnableParallel(
            context=itemgetter("question") | retriever | format_docs,
            question=itemgetter("question")
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    print("RAG chain initialized successfully")

    return RAGChain(rag_chain, retriever)
