import os
import pickle
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI
from ui import apply_custom_css, render_banner, render_sidebar, render_chat_history

# ------------------------- CONFIGURATION -------------------------
SECTIONS = {
    "All Sections": None,
    "Returns": "returns",
    "Delivery": "delivery",
    "Refunds": "refunds",
    "Sellers": "sellers",
    "Payments": "payments",
    "Customer Support": "customer_support",
}

FAISS_DIR = "./faiss_index"
INDEX_PATH = os.path.join(FAISS_DIR, "index.faiss")
METADATA_PATH = os.path.join(FAISS_DIR, "metadata.pkl")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Read directly from .streamlit/secrets.toml
LLM_MODEL = st.secrets.get("MODEL", "llama-3.3-70b-versatile")
API_KEY = st.secrets.get("API_KEY", "")
BASE_URL = st.secrets.get("BASE_URL", "https://api.groq.com/openai/v1")

TOP_K = 5
FETCH_K = 10

# ------------------------- PAGE CONFIG -------------------------
st.set_page_config(page_title="Daraz Support Assistant",
                   page_icon="🛍️", layout="centered")

apply_custom_css()
render_banner()

# ------------------------- LOAD RESOURCES (Cached) -------------------------


@st.cache_resource(show_spinner="Loading knowledge base...")
def load_index_and_metadata():
    # Check if files exist AND are not empty (0 bytes)
    if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH) or os.path.getsize(INDEX_PATH) == 0:
        return None, None
    try:
        index = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)
        return index, metadata
    except Exception:
        return None, None

    
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource(show_spinner=False)
def load_openai_client(api_key: str, base_url: str):
    if not api_key:
        return None

    try:
        # Initialize OpenAI-compatible client with Groq's base URL
        return OpenAI(api_key=api_key, base_url=base_url)
    except Exception as e:
        st.error(f"Failed to initialize LLM client: {e}")
        return None


# Resolve resources
index, metadata = load_index_and_metadata()
embedder = load_embedder()
openai_client = load_openai_client(API_KEY, BASE_URL)

# ------------------------- VALIDATION -------------------------
if index is None or metadata is None:
    st.error(
        f"⚠️ Couldn't find a valid pre-built index at `{FAISS_DIR}/`.\n\n"
        "The file may be missing, empty (0 bytes), or corrupted.\n"
        "Please run `python ingest.py` first, then **restart the Streamlit server**."
    )
    st.stop()

if not API_KEY:
    st.error(
        "⚠️ No API key found.\n\n"
        "Please add `API_KEY` to your `.streamlit/secrets.toml` file or set it as an environment variable."
    )
    st.stop()

if openai_client is None:
    st.error(
        "⚠️ Failed to initialize the LLM client. Check your API key and model configuration.")
    st.stop()


# ------------------------- SIDEBAR -------------------------
selected_label, selected_department = render_sidebar(SECTIONS)
with st.sidebar:
    if index is not None:
        st.caption(f"📊 Index size: {index.ntotal} chunks")
    else:
        st.caption("📊 Index size: Not loaded")
# ------------------------- RETRIEVAL ENGINE -------------------------


def retrieve(query: str, department: str | None, top_k: int = TOP_K) -> list[dict]:
    """Embeds query, searches FAISS, and optionally filters by department."""
    query_vec = embedder.encode(
        [query], convert_to_numpy=True).astype("float32")

    # Fetch more candidates if filtering is applied to ensure we get top_k valid results
    fetch_k = FETCH_K if department else top_k
    distances, indices = index.search(query_vec, fetch_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1 or idx >= len(metadata):
            continue
        record = metadata[idx]
        if department and record["department"] != department:
            continue
        results.append({**record, "score": float(dist)})
        if len(results) >= top_k:
            break
    return results


def build_context(chunks: list[dict]) -> str:
    """Formats retrieved chunks into a structured context string."""
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(
            f"[Source {i} | dept: {c['department']} | file: {c['source_file']}]\n{c['text']}")
    return "\n\n".join(parts)


SYSTEM_PROMPT = """You are the Daraz Customer Support & Operations Assistant.
Answer using ONLY the provided context chunks. 
Rules:
- If context lacks info, state clearly and suggest contacting Daraz Support. Do not guess.
- Keep answers concise, friendly, and practical.
- Mention the relevant department/policy when helpful.
- Never invent policy details, dates, or numbers."""


def generate_answer_stream(query: str, chunks: list[dict]):
    """Streams the LLM response based on retrieved context."""
    context = build_context(chunks)
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

    try:
        stream = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except Exception as e:
        yield f"⚠️ Error generating response: {str(e)}"


# ------------------------- CHAT INTERFACE -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

render_chat_history()

prompt = st.chat_input(
    "Ask about returns, delivery, refunds, sellers, payments...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        chunks = retrieve(prompt, selected_department, top_k=TOP_K)

        if not chunks:
            answer = f"I couldn't find anything relevant in **{selected_label}**. Try another section or **All Sections**."
            st.markdown(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer})
        else:
            # st.write_stream is the official, warning-free way to handle streaming in modern Streamlit
            full_response = st.write_stream(
                generate_answer_stream(prompt, chunks))

            with st.expander("📎 Sources"):
                for s in chunks:
                    st.markdown(
                        f"- **{s['department']}** — `{s['source_file']}`")

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": [{"department": s["department"], "source_file": s["source_file"]} for s in chunks]
            })
