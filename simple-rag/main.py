#!/usr/bin/env python3
"""Simple RAG Chat - Interactive chat with your documents using LangChain, ChromaDB, and Ollama."""

import sys
import argparse

from src.document_loader import load_documents
from src.vector_store import initialize_vector_store
from src.rag_chain import create_rag_chain
from src.chat import chat_loop


def main():
    """Main entry point for the RAG chat application."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Simple RAG Chat - Ask questions about your documents"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Force reload of documents and rebuild vector store"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3:8b",
        help="Ollama model to use (default: qwen3:8b)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="inputs",
        help="Directory containing input documents (default: inputs)"
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("Simple RAG Chat - Initializing...")
    print("=" * 70 + "\n")

    try:
        # Check if vector store exists
        import os
        vector_store_exists = os.path.exists("./chroma_db") and os.listdir("./chroma_db")

        # Step 1: Load documents (if needed)
        documents = None
        if args.reload or not vector_store_exists:
            print("Step 1: Loading documents...")
            documents = load_documents(args.input_dir)
            print()
        else:
            print("Step 1: Skipping document loading (using existing vector store)")
            print()

        # Step 2: Initialize vector store
        print("Step 2: Initializing vector store...")
        vector_store = initialize_vector_store(
            documents=documents,
            force_reload=args.reload
        )
        print()

        # Step 3: Create RAG chain
        print("Step 3: Creating RAG chain...")
        qa_chain = create_rag_chain(vector_store, model_name=args.model)
        print()

        # Step 4: Start chat loop
        print("Step 4: Starting chat interface...")
        print("=" * 70 + "\n")
        chat_loop(qa_chain)

    except ValueError as e:
        print(f"\nError: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}\n")
        print("Please check that:")
        print("1. Ollama is running: ollama serve")
        print(f"2. The model is installed: ollama pull {args.model}")
        print("3. Documents exist in the input directory")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()
