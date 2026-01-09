"""Streamlit app for comparing RAG approaches side-by-side."""

import time
import subprocess
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

from simple_rag_client import SimpleRAGClient
from lightrag_client import LightRAGClient
from prompts import SYSTEM_PROMPT
from ollama_client import ollama_chat


@st.cache_data(ttl=60)
def get_ollama_models():
    """Get list of locally available Ollama models."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")[1:]  # Skip header
            models = []
            for line in lines:
                if line.strip():
                    model_name = line.split()[0]
                    # Filter out embedding models
                    if "embed" not in model_name.lower():
                        models.append(model_name)
            return models if models else ["qwen3:1.7b"]
    except Exception:
        pass
    return ["qwen3:1.7b"]


# Initialize clients with model parameter
@st.cache_resource
def get_simple_rag_client(model_name: str):
    return SimpleRAGClient(model_name=model_name)


@st.cache_resource
def get_lightrag_client():
    return LightRAGClient()


# Page config
st.set_page_config(
    page_title="RAG Comparison",
    page_icon="🔍",
    layout="wide"
)

st.title("RAG Comparison")
st.markdown("Compare **Simple RAG** (vector similarity) vs **LightRAG** (knowledge graph)")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "last_results" not in st.session_state:
    st.session_state.last_results = None
if "last_query" not in st.session_state:
    st.session_state.last_query = None
if "last_mode" not in st.session_state:
    st.session_state.last_mode = None
if "last_errors" not in st.session_state:
    st.session_state.last_errors = {}
if "last_model" not in st.session_state:
    st.session_state.last_model = None

# Get available models
available_models = get_ollama_models()

# Sidebar for settings
with st.sidebar:
    st.header("Settings")

    # Global settings
    st.subheader("Global")
    show_sources = st.checkbox(
        "Show sources",
        value=True,
        help="Show the retrieved chunks/references that were used as evidence.",
    )

    st.markdown("---")

    # Simple RAG settings
    st.subheader("Simple RAG")
    simple_rag_model = st.selectbox(
        "Model",
        options=available_models,
        index=0,
        key="simple_rag_model",
        help="Model used for generating answers"
    )

    st.markdown("---")

    # LightRAG specific settings
    st.subheader("LightRAG")
    lightrag_model = st.selectbox(
        "Model",
        options=available_models,
        index=0,
        key="lightrag_model",
        help="Model for LightRAG (updates .env, requires server restart)"
    )

    lightrag_mode = st.selectbox(
        "Search mode",
        options=["naive", "hybrid", "local", "global", "mix"],
        index=0,
        help="""
        - **naive**: Simple vector search (most similar to Simple RAG)
        - **hybrid**: Combines local + global search
        - **local**: Entity-focused search
        - **global**: Relationship-focused search
        - **mix**: Adaptive mode selection
        """
    )

    lightrag_debug = st.checkbox(
        "Show retrieved context",
        value=False,
        help="Debug: show the full raw context string LightRAG would send to the LLM."
    )

    lightrag_strict = st.checkbox(
        "Strict: answer only from retrieved chunks",
        value=True,
        help="Safer: ignores LightRAG's generated answer and regenerates it locally from the retrieved chunks only (prevents leaking outside knowledge).",
    )

    lightrag_prompt_delivery = st.selectbox(
        "Prompt delivery",
        options=["user_prompt (recommended)", "inline into query (fallback)"],
        index=0,
        help=(
            "How we deliver the context-only instruction to the LightRAG server. "
            "If the server ignores `user_prompt`, use the fallback (it may slightly hurt retrieval)."
        ),
    )

    lightrag_evidence_gate = st.checkbox(
        "Evidence gate: require sources from allowed files",
        value=True,
        help=(
            "Keeps LightRAG's native answer, but refuses answers when no sources are returned "
            "or when sources are not from the allowed book(s)."
        ),
    )

    lightrag_allowed_files = st.text_input(
        "Allowed file(s) (comma-separated substring match)",
        value="harry-potter-and-the-philosophers-stone",
        help="Example: `philosophers-stone` (matches file paths that contain this text). Leave empty to allow any file.",
    )

    # Update LightRAG .env if model changed
    if st.button("Apply LightRAG Model", help="Updates .env file - restart server to apply"):
        try:
            from pathlib import Path
            env_path = Path(__file__).parent.parent / "lightrag" / ".env"
            env_content = env_path.read_text()

            # Replace LLM_MODEL line
            import re
            new_content = re.sub(
                r'^LLM_MODEL=.*$',
                f'LLM_MODEL={lightrag_model}',
                env_content,
                flags=re.MULTILINE
            )
            env_path.write_text(new_content)
            st.success(f"Updated to {lightrag_model}. Restart LightRAG server to apply.")
        except Exception as e:
            st.error(f"Failed to update: {e}")

# Get cached clients with selected model
simple_rag_client = get_simple_rag_client(simple_rag_model)
lightrag_client = get_lightrag_client()

# Initialize custom prompt in session state
if "custom_prompt" not in st.session_state:
    st.session_state.custom_prompt = SYSTEM_PROMPT

# Editable system prompt
with st.expander("System Prompt", expanded=False):
    custom_prompt = st.text_area(
        "Edit the system prompt sent to both RAG systems:",
        value=st.session_state.custom_prompt,
        height=150,
        key="prompt_editor"
    )
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Reset to default"):
            st.session_state.custom_prompt = SYSTEM_PROMPT
            st.rerun()
    # Update session state if changed
    if custom_prompt != st.session_state.custom_prompt:
        st.session_state.custom_prompt = custom_prompt

# Query input with form (allows Enter to submit)
# Note: we clear the input box on submit, but we always render the last question
# above the results so the user doesn't lose track of what was asked.
with st.form(key="query_form", clear_on_submit=True):
    query = st.text_input(
        "Enter your question:",
        placeholder="Ask something about your documents...",
    )
    submit = st.form_submit_button("Send", type="primary")


def query_simple_rag(client: SimpleRAGClient, question: str, system_prompt: str) -> tuple[dict, float]:
    """Query simple RAG and return result with timing."""
    start = time.time()
    result = client.query(question, system_prompt=system_prompt)
    elapsed = time.time() - start
    return result, elapsed


def query_lightrag(client: LightRAGClient, question: str, mode: str, debug: bool, system_prompt: str) -> tuple[dict, float]:
    """Query LightRAG and return result with timing."""
    start = time.time()
    result = client.query(question, mode=mode, return_context=debug, system_prompt=system_prompt)
    elapsed = time.time() - start
    return result, elapsed


def _parse_allowed_file_substrings(value: str) -> list[str]:
    if not value:
        return []
    parts = [p.strip() for p in value.split(",")]
    return [p for p in parts if p]


def _filter_sources_by_allowed_files(sources: list[dict], allowed: list[str]) -> list[dict]:
    if not allowed:
        return sources
    out: list[dict] = []
    for s in sources:
        meta = s.get("metadata") or {}
        fp = str(meta.get("file", "") or "")
        if any(a.lower() in fp.lower() for a in allowed):
            out.append(s)
    return out


def strict_answer_from_sources(
    *,
    question: str,
    sources: list[dict],
    model: str,
    system_prompt: str,
) -> str:
    """
    Generate an answer strictly from retrieved sources.
    If there is no evidence, return the standard "I don't know..." response.
    """
    if not sources:
        return "I don't know based on the provided context"

    # Build a context block from full chunk contents if available.
    chunks: list[str] = []
    for src in sources:
        raw = src.get("raw_content") or src.get("content") or ""
        if isinstance(raw, str) and raw.strip():
            chunks.append(raw.strip())

    if not chunks:
        return "I don't know based on the provided context"

    # Keep context reasonably bounded.
    context = "\n\n---\n\n".join(chunks)
    context = context[:12000]

    user_prompt = f"""CONTEXT (retrieved passages):
{context}

QUESTION:
{question}
"""
    try:
        answer = ollama_chat(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            timeout_s=180.0,
        )
        return answer.strip() or "I don't know based on the provided context"
    except Exception:
        # If strict generation fails, fall back to refusing rather than leaking knowledge.
        return "I don't know based on the provided context"


def render_result(container, name, result_data, error, show_sources, model_info=None, show_context=False):
    """Render a single RAG result."""
    with container:
        if name == "simple":
            model_short = model_info.split(":")[0] if model_info and ":" in model_info else (model_info or "")
            st.subheader(f"Simple RAG ({model_short})")
        else:
            st.subheader(f"LightRAG ({model_info})")

        if error:
            st.error(f"Error: {error}")
        elif result_data:
            st.success(f"Response time: {result_data['time']:.2f}s")
            st.markdown(result_data["result"]["answer"])

            # Show retrieved context (debug mode)
            if show_context:
                ctx = result_data["result"].get("context")
                if ctx:
                    with st.expander("Retrieved Context (sent to LLM)", expanded=False):
                        st.text(ctx[:3000])
                        if len(ctx) > 3000:
                            st.caption("... (truncated)")
                else:
                    st.caption("Context was not fetched for this run. Re-run the query with “Show retrieved context” enabled.")

            if show_sources and result_data["result"].get("sources"):
                with st.expander("Sources", expanded=False):
                    for i, src in enumerate(result_data["result"]["sources"], 1):
                        st.markdown(f"**Source {i}**")
                        st.text(src["content"])
                        if src.get("metadata"):
                            file_key = "filename" if name == "simple" else "file"
                            st.caption(f"File: {src['metadata'].get(file_key, 'unknown')}")
            elif show_sources and name == "light":
                st.caption("No sources returned by LightRAG for this query (answer may be unreliable unless strict mode is enabled).")
        else:
            st.info("Querying...")


# Question banner should appear ABOVE the result columns (including "Querying...").
question_banner = st.container()

# Create columns for results
left_col, right_col = st.columns(2)

if submit and query:
    # Persist the question immediately so it's visible even after the form clears.
    st.session_state.last_query = query

    with question_banner:
        st.markdown(f"**Question:** {query}")

    # Create placeholders
    left_placeholder = left_col.empty()
    right_placeholder = right_col.empty()

    # Show loading state
    with left_placeholder.container():
        st.subheader(f"Simple RAG ({simple_rag_model.split(':')[0]})")
        st.info("Querying...")

    with right_placeholder.container():
        st.subheader(f"LightRAG ({lightrag_mode})")
        st.info("Querying...")

    # Query both systems in parallel
    results = {}
    errors = {}

    # Get current prompt from session state
    current_prompt = st.session_state.custom_prompt

    with ThreadPoolExecutor(max_workers=2) as executor:
        # Decide how we deliver the prompt to LightRAG.
        # Fallback option injects the instruction into the query text itself.
        lightrag_query_text = (
            f"{current_prompt}\n\nQUESTION:\n{query}"
            if lightrag_prompt_delivery.startswith("inline")
            else query
        )
        lightrag_prompt_text = "" if lightrag_prompt_delivery.startswith("inline") else current_prompt

        futures = {
            executor.submit(query_simple_rag, simple_rag_client, query, current_prompt): "simple",
            executor.submit(query_lightrag, lightrag_client, lightrag_query_text, lightrag_mode, lightrag_debug, lightrag_prompt_text): "light"
        }

        for future in as_completed(futures):
            name = futures[future]
            try:
                result, elapsed = future.result()

                # Optionally override LightRAG answer with strict context-only generation.
                if name == "light" and lightrag_strict:
                    strict = strict_answer_from_sources(
                        question=query,
                        sources=result.get("sources") or [],
                        model=lightrag_model,
                        system_prompt=current_prompt,
                    )
                    result["answer"] = strict
                elif name == "light" and lightrag_evidence_gate:
                    allowed = _parse_allowed_file_substrings(lightrag_allowed_files)
                    filtered = _filter_sources_by_allowed_files(result.get("sources") or [], allowed)
                    result["sources"] = filtered
                    if not filtered:
                        result["answer"] = "I don't know based on the provided context"

                results[name] = {"result": result, "time": elapsed}
            except Exception as e:
                errors[name] = str(e)

            # Update display immediately when each result arrives
            if name == "simple":
                with left_placeholder.container():
                    render_result(
                        st.container(), "simple",
                        results.get("simple"), errors.get("simple"),
                        show_sources, simple_rag_model
                    )
            else:
                with right_placeholder.container():
                    render_result(
                        st.container(), "light",
                        results.get("light"), errors.get("light"),
                        show_sources, lightrag_mode, show_context=lightrag_debug
                    )

    # Save results to session state
    st.session_state.last_results = results
    st.session_state.last_mode = lightrag_mode
    st.session_state.last_model = simple_rag_model
    st.session_state.last_errors = errors

    # Save to history
    st.session_state.history.append({
        "query": query,
        "results": results
    })

# Display last results (persists when settings change)
elif st.session_state.last_results:
    results = st.session_state.last_results
    errors = st.session_state.last_errors

    if st.session_state.last_query:
        with question_banner:
            st.markdown(f"**Question:** {st.session_state.last_query}")

    with left_col:
        render_result(
            st.container(), "simple",
            results.get("simple"), errors.get("simple"),
            show_sources, st.session_state.last_model or simple_rag_model
        )

    with right_col:
        render_result(
            st.container(), "light",
            results.get("light"), errors.get("light"),
            show_sources, st.session_state.last_mode or lightrag_mode,
            show_context=lightrag_debug
        )

# Show history
if st.session_state.history:
    st.markdown("---")
    st.subheader("Previous Queries")

    for i, item in enumerate(reversed(st.session_state.history[:-1])):  # Skip the current one
        # Truncate for expander title
        title_query = item['query'][:50] + "..." if len(item['query']) > 50 else item['query']
        with st.expander(f"Q: {title_query}"):
            # Show full query
            st.markdown(f"**Query:** {item['query']}")
            st.markdown("---")

            hist_left, hist_right = st.columns(2)

            with hist_left:
                st.markdown("**Simple RAG**")
                if "simple" in item["results"]:
                    st.caption(f"Time: {item['results']['simple']['time']:.2f}s")
                    answer = item["results"]["simple"]["result"]["answer"]
                    st.markdown(answer[:500] + "..." if len(answer) > 500 else answer)

            with hist_right:
                st.markdown("**LightRAG**")
                if "light" in item["results"]:
                    st.caption(f"Time: {item['results']['light']['time']:.2f}s")
                    answer = item["results"]["light"]["result"]["answer"]
                    st.markdown(answer[:500] + "..." if len(answer) > 500 else answer)
