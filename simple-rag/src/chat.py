"""Interactive chat loop for RAG system."""

from typing import Dict, Any


class ChatSession:
    """Manages the interactive chat session."""

    def __init__(self, qa_chain):
        """
        Initialize chat session.

        Args:
            qa_chain: Configured RAG chain
        """
        self.qa_chain = qa_chain
        self.show_sources = False

    def display_welcome(self) -> None:
        """Display welcome message and instructions."""
        print("\n" + "=" * 70)
        print("Welcome to Simple RAG Chat!")
        print("=" * 70)
        print("\nAsk questions about your documents and get AI-powered answers.")
        print("\nCommands:")
        print("  /exit or /quit  - Exit the chat")
        print("  /help           - Show this help message")
        print("  /sources        - Toggle source document display")
        print("\nType your question and press Enter to get started!")
        print("=" * 70 + "\n")

    def display_help(self) -> None:
        """Display help information."""
        print("\nAvailable commands:")
        print("  /exit, /quit    - Exit the chat")
        print("  /help           - Show this help message")
        print("  /sources        - Toggle source document display")
        print(f"\nSource display is currently: {'ON' if self.show_sources else 'OFF'}")
        print()

    def toggle_sources(self) -> None:
        """Toggle source document display."""
        self.show_sources = not self.show_sources
        status = "enabled" if self.show_sources else "disabled"
        print(f"\nSource document display {status}\n")

    def format_sources(self, source_documents: list) -> str:
        """
        Format source documents for display.

        Args:
            source_documents: List of source Document objects

        Returns:
            Formatted string with source information
        """
        if not source_documents:
            return ""

        output = "\n" + "-" * 70 + "\n"
        output += "Sources:\n"
        output += "-" * 70 + "\n"

        for idx, doc in enumerate(source_documents, 1):
            # Handle both Document objects and dicts
            if hasattr(doc, 'metadata'):
                filename = doc.metadata.get("filename", "Unknown")
                chunk_idx = doc.metadata.get("chunk_index", "?")
                content = doc.page_content
            elif isinstance(doc, dict):
                filename = doc.get("metadata", {}).get("filename", "Unknown")
                chunk_idx = doc.get("metadata", {}).get("chunk_index", "?")
                content = doc.get("page_content", "")
            else:
                filename = "Unknown"
                chunk_idx = "?"
                content = str(doc)

            content_preview = str(content)[:150].replace("\n", " ")

            output += f"\n[{idx}] {filename} (chunk {chunk_idx})\n"
            output += f"    {content_preview}...\n"

        return output

    def process_query(self, query: str) -> None:
        """
        Process a user query and display results.

        Args:
            query: User's question
        """
        try:
            print("\nThinking...\n")
            result = self.qa_chain({"query": query})

            # Display answer
            answer = result.get("result", "No answer generated")
            print(answer)
            print()

            # Display sources if enabled
            if self.show_sources and "source_documents" in result:
                print(self.format_sources(result["source_documents"]))

        except Exception as e:
            import traceback
            print(f"\nError processing question: {e}")
            print("\nDebug traceback:")
            traceback.print_exc()
            print("\nPlease try rephrasing your question or check your connection to Ollama.\n")

    def handle_command(self, command: str) -> bool:
        """
        Handle special commands.

        Args:
            command: Command string (without leading /)

        Returns:
            True if should exit chat, False otherwise
        """
        command = command.lower()

        if command in ["exit", "quit"]:
            return True
        elif command == "help":
            self.display_help()
        elif command == "sources":
            self.toggle_sources()
        else:
            print(f"\nUnknown command: /{command}")
            print("Type /help for available commands.\n")

        return False

    def run(self) -> None:
        """Run the interactive chat loop."""
        self.display_welcome()

        while True:
            try:
                # Get user input
                user_input = input("> ").strip()

                # Skip empty input
                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    command = user_input[1:]
                    if self.handle_command(command):
                        break
                    continue

                # Process query
                self.process_query(user_input)

            except KeyboardInterrupt:
                print("\n\nInterrupted by user.")
                break
            except EOFError:
                print("\n\nEnd of input.")
                break
            except Exception as e:
                print(f"\nUnexpected error: {e}\n")
                continue

        print("\nThank you for using Simple RAG Chat. Goodbye!\n")


def chat_loop(qa_chain) -> None:
    """
    Start an interactive chat session.

    Args:
        qa_chain: Configured RAG chain
    """
    session = ChatSession(qa_chain)
    session.run()
