from __future__ import annotations

import io
import json
import os
import tempfile
import zipfile
from pathlib import Path

import streamlit as st
from retriever import hybrid_search

st.set_page_config(page_title="RepoLens Chat", page_icon="🔎", layout="wide")
st.title("🔎 RepoLens — Codebase Chat")
st.caption("Upload a project ZIP or source files, ask questions, and get answers grounded in retrieved code.")

IGNORED_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", "dist", "build"}
ALLOWED_SUFFIXES = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs", ".cpp", ".h",
    ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".html", ".css", ".sql",
    ".sh", ".c", ".cs", ".php", ".rb", ".swift", ".kt"
}
MAX_FILE_BYTES = 1_000_000
MAX_TOTAL_BYTES = 30_000_000
MAX_CONTEXT_CHARS = 12_000


def is_local_ollama_url(base_url: str) -> bool:
    host = base_url.lower()
    return "localhost" in host or "127.0.0.1" in host or "0.0.0.0" in host


def trim_for_model(text: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def safe_name(name: str) -> bool:
    p = Path(name)
    return not p.is_absolute() and ".." not in p.parts and not any(part in IGNORED_DIRS for part in p.parts)

def write_uploads(uploaded_files, destination: Path):
    total = 0
    count = 0
    for uploaded in uploaded_files or []:
        name = uploaded.name.replace("\\", "/")
        data = uploaded.getvalue()
        if name.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as archive:
                    for member in archive.infolist():
                        member_name = member.filename.replace("\\", "/")
                        if member.is_dir() or not safe_name(member_name):
                            continue
                        suffix = Path(member_name).suffix.lower()
                        if suffix not in ALLOWED_SUFFIXES or member.file_size > MAX_FILE_BYTES:
                            continue
                        total += member.file_size
                        if total > MAX_TOTAL_BYTES:
                            raise ValueError("Upload exceeds the 30 MB extracted-text limit.")
                        target = destination / member_name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(archive.read(member))
                        count += 1
            except zipfile.BadZipFile:
                st.error(f"{uploaded.name} is not a valid ZIP file.")
        else:
            if not safe_name(name) or Path(name).suffix.lower() not in ALLOWED_SUFFIXES:
                continue
            if len(data) > MAX_FILE_BYTES:
                continue
            total += len(data)
            if total > MAX_TOTAL_BYTES:
                raise ValueError("Upload exceeds the 30 MB text limit.")
            target = destination / Path(name).name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            count += 1
    return count

def call_chat_api(messages, api_key, base_url, model):
    import urllib.request

    endpoint = base_url.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint = endpoint + "/chat/completions"

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            data = json.load(response)
        return data["choices"][0]["message"]["content"]
    except Exception as exc:
        raise RuntimeError(f"LLM request failed: {exc}") from exc

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("API key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    base_url = st.text_input("API base URL", value=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"))
    model = st.text_input("Model", value=os.getenv("OPENAI_MODEL", "llama3.2:latest"))
    semantic = st.toggle("Use semantic embeddings + RRF", value=False)
    st.caption("For local Ollama use: API base URL like http://localhost:11434/v1 and model like llama3.2:latest. No API key is needed for local Ollama.")

st.subheader("1. Upload your codebase")
uploads = st.file_uploader(
    "Upload a project ZIP or individual source files",
    type=["zip", "py", "js", "jsx", "ts", "tsx", "java", "go", "rs", "cpp", "h", "md", "txt", "json", "yaml", "yml", "toml", "html", "css", "sql", "sh", "c", "cs", "php", "rb", "swift", "kt"],
    accept_multiple_files=True,
    help="ZIP uploads are unpacked safely. Common source, documentation, and configuration files are indexed; dependencies and binaries are skipped."
)

if "repo_dir" not in st.session_state:
    st.session_state.repo_dir = None
if "repo_temp" not in st.session_state:
    st.session_state.repo_temp = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "loaded_signature" not in st.session_state:
    st.session_state.loaded_signature = None

if st.button("Index uploaded files", type="primary", disabled=not uploads):
    tmp = tempfile.TemporaryDirectory(prefix="repolens_upload_")
    dest = Path(tmp.name)
    try:
        n = write_uploads(uploads, dest)
        if n == 0:
            st.error("No supported text/code files found in the upload.")
        else:
            if st.session_state.repo_temp:
                st.session_state.repo_temp.cleanup()
            st.session_state.repo_temp = tmp
            st.session_state.repo_dir = str(dest)
            st.session_state.loaded_signature = tuple((f.name, len(f.getvalue())) for f in uploads)
            st.session_state.messages = []
            st.success(f"Indexed {n} files. Ask a question below.")
    except Exception as exc:
        tmp.cleanup()
        st.error(str(exc))

if st.session_state.repo_dir:
    repo_path = Path(st.session_state.repo_dir)
    files = [p for p in repo_path.rglob("*") if p.is_file()]
    st.success(f"Codebase ready: {len(files)} files available for retrieval.")
    with st.expander("Show indexed files"):
        st.code("\n".join(str(p.relative_to(repo_path)) for p in files[:300]) or "No files")

    st.subheader("2. Chat with your code")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Retrieved code evidence"):
                    for item in message["sources"]:
                        st.markdown(f"**{item['path']} — lines {item['start_line']}-{item['end_line']}**")
                        st.code(item["text"], language="text")

    prompt = st.chat_input("Ask about your codebase, e.g. Explain the login flow")
    if prompt:
        if not api_key and not is_local_ollama_url(base_url):
            st.warning("Enter your model API key in the sidebar first. The chatbot needs a real LLM to answer arbitrary questions.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            try:
                with st.spinner("Searching your code and generating an answer..."):
                    retrieved = hybrid_search(repo_path, prompt, top_k=7, semantic=semantic)
                    results = retrieved.get("results", [])
                    if not results:
                        context = "No matching code chunks were retrieved."
                    else:
                        context_parts = []
                        for i, item in enumerate(results, 1):
                            context_parts.append(
                                f"[Source {i}: {item['path']} lines {item['start_line']}-{item['end_line']}]\n{trim_for_model(item['text'], 2500)}"
                            )
                        context = trim_for_model("\n\n---\n\n".join(context_parts), MAX_CONTEXT_CHARS)
                    system = (
                        "You are RepoLens, a careful software repository assistant. Answer the user's question "
                        "using the supplied repository evidence. Do not claim to have inspected files that are not "
                        "in the evidence. If evidence is insufficient, say what is missing. Cite relevant evidence "
                        "using exact paths and line ranges in the format `path:line-line`. Explain code clearly."
                    )
                    history = [{"role": "system", "content": system}]
                    for old in st.session_state.messages[-7:-1]:
                        history.append({"role": old["role"], "content": old["content"]})
                    history.append({
                        "role": "user",
                        "content": f"Question: {prompt}\n\nRetrieved repository evidence:\n{context}"
                    })
                    answer = call_chat_api(history, api_key, base_url, model)
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": results})
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    if results:
                        with st.expander("Retrieved code evidence"):
                            for item in results:
                                st.markdown(f"**{item['path']} — lines {item['start_line']}-{item['end_line']}**")
                                st.code(item["text"], language="text")
            except Exception as exc:
                st.error(str(exc))
else:
    st.info("Upload a ZIP of your project or select source files, then click **Index uploaded files**.")
    st.markdown("**Example questions**")
    st.markdown("- Explain the overall architecture of this repository.\n- Where is authentication implemented?\n- Trace how a request moves from the API route to the database.\n- Find the function that handles a particular feature.")
