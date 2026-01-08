"""Document loading and text splitting utilities."""

import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def load_documents(input_dir: str = "inputs") -> List[Document]:
    """
    Load and split documents from the input directory.

    Args:
        input_dir: Directory containing text files to load

    Returns:
        List of Document objects with text chunks and metadata

    Raises:
        ValueError: If input directory doesn't exist or is empty
    """
    input_path = Path(input_dir)

    # Validate input directory
    if not input_path.exists():
        raise ValueError(f"Input directory '{input_dir}' does not exist")

    # Find all .txt files
    txt_files = list(input_path.glob("*.txt"))
    if not txt_files:
        raise ValueError(f"No .txt files found in '{input_dir}'")

    print(f"Found {len(txt_files)} text file(s) to process...")

    # Load all documents
    all_documents = []
    for txt_file in txt_files:
        try:
            print(f"Loading: {txt_file.name}")
            loader = TextLoader(str(txt_file), encoding="utf-8")
            documents = loader.load()

            # Add source filename to metadata
            for doc in documents:
                doc.metadata["filename"] = txt_file.name
                doc.metadata["source_path"] = str(txt_file)

            all_documents.extend(documents)
        except Exception as e:
            print(f"Warning: Failed to load {txt_file.name}: {e}")
            continue

    if not all_documents:
        raise ValueError("No documents could be loaded successfully")

    print(f"Loaded {len(all_documents)} document(s)")

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    print("Splitting documents into chunks...")
    chunks = text_splitter.split_documents(all_documents)

    # Add chunk index to metadata
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx

    print(f"Created {len(chunks)} chunks")

    return chunks
