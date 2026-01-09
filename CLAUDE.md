# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project compares two RAG (Retrieval-Augmented Generation) approaches:

- **simple-rag/**: Traditional RAG using vector similarity search (LangChain + ChromaDB)
- **lightrag/**: GraphRAG implementation using knowledge graphs (LightRAG library + Ollama)

The goal is to evaluate differences in retrieval quality and answer accuracy between traditional vector-based RAG and graph-based RAG approaches.

## Commands

### simple-rag (Traditional RAG)

```bash
cd simple-rag

# Install dependencies
uv sync

# Run the chat application
python main.py

# Force reload documents (rebuild vector database)
python main.py --reload

# Use different Ollama model
python main.py --model llama3.2
```

### lightrag (GraphRAG)

```bash
cd lightrag

# Install dependencies
uv sync

# Run (currently stub)
python main.py
```

### Prerequisites

- Python 3.13+
- Ollama running locally (`ollama serve`)
- qwen3:8b model installed (`ollama pull qwen3:8b`)

## Architecture

### simple-rag (Traditional RAG)

Four-stage pipeline:

1. **Document Loading** (`src/document_loader.py`): Loads `.txt` files, splits into ~1000 char chunks with 200 char overlap
2. **Vector Store** (`src/vector_store.py`): ChromaDB with `all-MiniLM-L6-v2` embeddings, persists to `chroma_db/`
3. **RAG Chain** (`src/rag_chain.py`): LangChain LCEL chain retrieving top 4 similar chunks, passes to Ollama LLM
4. **Chat Interface** (`src/chat.py`): Interactive CLI with `/help`, `/sources`, `/exit` commands

### lightrag (GraphRAG)

Uses LightRAG library which builds a knowledge graph from documents, storing:
- Entities and relationships extracted by LLM
- Graph structure for traversal-based retrieval
- Storage in `rag_storage/` directory (JSON-based key-value stores)

### Key Differences

| Aspect | simple-rag | lightrag |
|--------|------------|----------|
| Retrieval | Vector similarity (embedding distance) | Graph traversal + entity relationships |
| Storage | ChromaDB vector database | JSON-based graph storage |
| Context | Independent text chunks | Connected entity/relationship context |
