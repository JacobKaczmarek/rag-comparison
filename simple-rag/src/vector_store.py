"""Vector store initialization and management using ChromaDB."""

import os
from typing import List, Optional

import chromadb
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


def get_embeddings():
    """
    Initialize the embedding model.

    Returns:
        HuggingFaceEmbeddings configured with all-MiniLM-L6-v2
    """
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def initialize_vector_store(
    documents: Optional[List[Document]] = None,
    persist_directory: str = "./chroma_db",
    collection_name: str = "documents",
    force_reload: bool = False
) -> Chroma:
    """
    Initialize or load the ChromaDB vector store.

    Args:
        documents: Documents to add to the store (required if force_reload or first run)
        persist_directory: Directory to store ChromaDB data
        collection_name: Name of the collection
        force_reload: If True, recreate the vector store from scratch

    Returns:
        Chroma vector store instance

    Raises:
        ValueError: If vector store doesn't exist and no documents provided
    """
    embeddings = get_embeddings()

    # Check if vector store already exists
    store_exists = os.path.exists(persist_directory) and os.listdir(persist_directory)

    if force_reload and store_exists:
        print("Force reload enabled. Deleting existing vector store...")
        import shutil
        shutil.rmtree(persist_directory)
        store_exists = False

    if store_exists:
        print("Loading existing vector store...")
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory
        )

        # Verify the store has documents
        collection_count = vector_store._collection.count()
        if collection_count > 0:
            print(f"Loaded vector store with {collection_count} embeddings")
            return vector_store
        else:
            print("Vector store exists but is empty. Will rebuild...")
            store_exists = False

    # Create new vector store
    if not documents:
        raise ValueError(
            "No existing vector store found and no documents provided. "
            "Please provide documents to initialize the vector store."
        )

    print(f"Creating new vector store with {len(documents)} documents...")
    print("This may take a few minutes on first run...")

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory
    )

    print(f"Vector store created with {len(documents)} embeddings")
    print(f"Persisted to: {persist_directory}")

    return vector_store


def reset_vector_store(persist_directory: str = "./chroma_db") -> None:
    """
    Delete the vector store to force re-indexing.

    Args:
        persist_directory: Directory containing ChromaDB data
    """
    if os.path.exists(persist_directory):
        import shutil
        shutil.rmtree(persist_directory)
        print(f"Vector store deleted: {persist_directory}")
    else:
        print("No vector store found to delete")
