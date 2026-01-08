# Simple RAG Chat

A simple Retrieval-Augmented Generation (RAG) chat application using LangChain, ChromaDB, and Ollama. Ask questions about your documents and get AI-powered answers based on the content.

## Features

- **Local LLM**: Uses Ollama with qwen3:8b model for private, offline AI chat
- **Vector Database**: ChromaDB for efficient document retrieval with persistence
- **Document Processing**: Automatic text splitting and embedding generation
- **Interactive CLI**: User-friendly command-line chat interface
- **Source Attribution**: Optional display of source document chunks

## Prerequisites

- Python 3.13 or higher
- [Ollama](https://ollama.ai/) installed and running
- uv package manager (recommended) or pip

## Installation

1. Clone this repository or navigate to the project directory

2. Install dependencies:
```bash
uv sync
```

Or with pip:
```bash
pip install -e .
```

3. Make sure Ollama is running:
```bash
ollama serve
```

4. Pull the qwen3:8b model:
```bash
ollama pull qwen3:8b
```

## Usage

### Basic Usage

Place your text files in the `inputs/` folder, then run:

```bash
python main.py
```

On first run, the application will:
1. Load documents from the `inputs/` folder
2. Split them into chunks
3. Generate embeddings using sentence-transformers
4. Store them in ChromaDB (persisted to `chroma_db/` folder)

Subsequent runs will use the cached vector database for faster startup.

### Command Line Options

```bash
# Use a different Ollama model
python main.py --model llama3.2

# Force reload of documents (rebuild vector database)
python main.py --reload

# Use a different input directory
python main.py --input-dir /path/to/documents
```

### Chat Commands

While in the chat:

- `/help` - Show available commands
- `/sources` - Toggle display of source documents
- `/exit` or `/quit` - Exit the chat
- `Ctrl+C` - Exit the chat

## Project Structure

```
simple-rag/
├── main.py                 # Entry point
├── src/
│   ├── __init__.py
│   ├── document_loader.py  # Document loading and text splitting
│   ├── vector_store.py     # ChromaDB setup and management
│   ├── rag_chain.py        # LangChain RAG chain configuration
│   └── chat.py             # Interactive chat loop
├── inputs/                 # Place your documents here
├── chroma_db/             # Vector database (auto-created)
└── pyproject.toml         # Project dependencies
```

## How It Works

1. **Document Loading**: Text files are loaded from the `inputs/` folder and split into chunks of ~1000 characters with 200-character overlap

2. **Embedding Generation**: Each chunk is embedded using the `all-MiniLM-L6-v2` model from sentence-transformers

3. **Vector Storage**: Embeddings are stored in ChromaDB with persistence to disk

4. **Retrieval**: When you ask a question, the system finds the 4 most relevant document chunks

5. **Generation**: The relevant chunks are passed to Ollama (qwen3:8b) along with your question to generate an answer

## Example Questions

Based on the included Harry Potter text:

- "Where does Harry Potter live?"
- "Who are the Dursleys?"
- "What happens at the beginning of the story?"

## Troubleshooting

**Error: Cannot connect to Ollama**
- Make sure Ollama is running: `ollama serve`
- Verify the model is installed: `ollama pull qwen3:8b`

**Error: No documents found**
- Ensure you have `.txt` files in the `inputs/` folder

**Slow first run**
- First run needs to generate embeddings for all documents, which can take a few minutes
- Subsequent runs use the cached database and are much faster

**Want to re-index documents**
- Use the `--reload` flag: `python main.py --reload`

## License

MIT

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) - RAG framework
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Ollama](https://ollama.ai/) - Local LLM runtime
- [sentence-transformers](https://www.sbert.net/) - Embedding models