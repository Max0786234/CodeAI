# RepoLens — AI Codebase Assistant

RepoLens is a small, explainable repository Q&A agent. It investigates a codebase using controlled tools and retrieves relevant code before answering. The project intentionally stays compact enough to understand and discuss in a fresher software engineering interview.

## Features

- **Function/tool calling:** the LLM can request allowlisted tools for listing, reading, searching, and hybrid code retrieval.
- **Structured reports:** `--report` emits a JSON report with findings, evidence, confidence, and a next step.
- **BM25 keyword retrieval:** built-in lightweight implementation; no third-party package required.
- **Semantic embeddings (optional):** sentence-transformers embeds code chunks and queries.
- **RRF (optional semantic mode):** Reciprocal Rank Fusion combines BM25 and vector rankings without comparing their raw scores.
- **Evidence-first workflow:** retrieved chunks include relative file paths and line ranges.

## Quick start (no API key)

Python 3.10+ recommended. From this folder:

```bash
python agent.py "Where is the tool registry?" --demo --trace
python -m unittest -v
```

Demo mode uses a deterministic fake model to demonstrate the agent loop. It does not demonstrate real LLM reasoning.

## Real LLM mode

Set an OpenAI-compatible API key and run:

```powershell
$env:OPENAI_API_KEY = "your-key"
python agent.py "Find where the agent loop is implemented"
```

Optional environment variables: `OPENAI_BASE_URL` (defaults to `https://api.openai.com/v1`) and `OPENAI_MODEL` (defaults to `gpt-4o-mini`).

## Retrieval

Keyword hybrid retrieval is available through the `hybrid_code_search` tool and can be tested directly:

```bash
python -c "from retriever import hybrid_search; import json; print(json.dumps(hybrid_search('.', 'agent tool registry'), indent=2))"
```

Enable semantic embeddings and RRF by installing the optional requirements:

```bash
pip install -r requirements.txt
```

Then ask the real agent to call `hybrid_code_search` with `semantic: true`, or call it directly:

```bash
python -c "from retriever import hybrid_search; import json; print(json.dumps(hybrid_search('.', 'where does the agent call tools?', semantic=True), indent=2))"
```

The first semantic run downloads `sentence-transformers/all-MiniLM-L6-v2`; model download and inference require internet access initially and sufficient RAM. Embeddings are computed at query time in this compact learning version; persistent vector indexing is intentionally out of scope.

## Architecture

1. `agent.py` defines the system prompt, tool schemas, controlled registry, model API call, agent loop, trace, and JSON report.
2. `retriever.py` chunks source files, computes BM25 rankings, optionally computes embedding similarity, and fuses rankings with RRF.
3. `test_agent.py` tests filesystem safety, tool contracts, agent traces, BM25 retrieval, and RRF behavior.

## Interview talking points

- BM25 handles exact identifiers and keywords; embeddings help find conceptually related code.
- RRF combines ranked lists using `1 / (k + rank)` and avoids assuming BM25 and cosine similarity scores share a scale.
- Chunk metadata preserves file path and line ranges for evidence and citations.
- Current limitations: no persistent vector index, no reranker, no distributed services, and no production deployment. Semantic search is optional; lexical search works without extra dependencies.


## Chatbot web interface

RepoLens now includes a Streamlit chatbot. It lets you upload a project ZIP or individual code files, index them, and ask questions in a chat interface. Answers are grounded in retrieved chunks and show the evidence.

### Run the chatbot

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown by Streamlit (usually `http://localhost:8501`).

1. Enter your OpenAI-compatible API key in the sidebar (or set `OPENAI_API_KEY` before launching).
2. Optionally set `OPENAI_BASE_URL` and `OPENAI_MODEL` for a compatible provider.
3. Upload a project ZIP or source files and click **Index uploaded files**.
4. Ask questions in the chat box.

The app safely skips common dependency/build folders and non-source files from ZIP uploads. It has a 30 MB extracted text limit and a 1 MB per-file limit. Semantic mode is optional and uses the embedding dependencies in `requirements.txt`.

Uploaded files are processed locally in the running app's temporary directory. Do not upload secrets or files you do not have permission to share. The selected LLM provider receives the retrieved code excerpts included in each question.
