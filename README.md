# RepoLens — Local AI Codebase Assistant

RepoLens is a lightweight Python project that helps you ask questions about a codebase using retrieval-first AI. Instead of sending the entire repository to a model blindly, it:

- chunks source files into searchable snippets,
- retrieves the most relevant code using BM25 and optional embedding search,
- combines rankings with Reciprocal Rank Fusion (RRF),
- sends only the relevant evidence to the model,
- returns answers grounded in real file paths and line ranges.

This keeps the project simple, explainable, and easy to run locally.

## What this project is doing

This project is a small repository assistant for software engineering tasks:

- find the file that implements a feature,
- answer questions about code structure,
- inspect logic in a repo without reading everything manually,
- work with uploaded ZIPs or local source folders,
- answer with evidence, not vague guesses.

The main idea is: retrieval quality matters more than raw model size for local code-analysis workflows.

## Why we used Llama locally

We used Llama through Ollama because it is a strong fit for this type of project:

- local and private: no secret API key required for local use,
- fast to prototype: easy to run on a dev machine,
- low friction: works well for learning and testing retrieval ideas,
- cheaper for development: no cloud billing while experimenting,
- practical for codebase Q&A: the model is not the main bottleneck; good retrieval is.

In other words, the real value here is the retrieval pipeline and evidence grounding, not just the LLM vendor. Local Llama is a good default for research, demos, and local validation.

The app is configured to work with:

- Ollama base URL: `http://localhost:11434/v1`
- model: `llama3.2:latest`

This means you can run the app locally without OpenAI credentials when using Ollama.

## Features

- local code indexing from files or uploaded ZIPs,
- BM25 keyword search,
- optional semantic embeddings via `sentence-transformers`,
- RRF fusion between lexical and semantic rankings,
- evidence display with file paths and line ranges,
- Streamlit web UI for chat-based repository Q&A,
- agent-style tool usage in `agent.py` for CLI-based code exploration.

## Project structure

```text
.
├── README.md
├── requirements.txt
├── app.py
├── agent.py
├── retriever.py
├── test_agent.py
├── .gitignore
└── .venv/                # local virtual environment, not committed
```

Important note: the local ZIP used for testing is intentionally not committed to GitHub. It stays on your machine for validation and is ignored by Git.

## Quick start

### 1) Create a virtual environment

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Start Ollama and pull the model

```bash
ollama pull llama3.2:latest
```

Then make sure Ollama is running locally.

### 4) Run the Streamlit app

```bash
streamlit run app.py
```

Open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

### 5) Upload a project or ZIP and ask a question

Examples:

- “Where is the binary search implementation?”
- “Show the two_sum function”
- “Which file contains the Fibonacci logic?”
- “Explain the login flow in this codebase”

## CLI agent mode

You can also run the non-UI tool-based agent:

```bash
python agent.py "Where is the tool registry?" --demo --trace
```

This is useful for testing the agent loop and report generation without the web interface.

## Retrieval internals

The core retrieval file is `retriever.py`.

It does the following:

1. splits code into chunks,
2. indexes file paths and content,
3. computes BM25 score for keyword matches,
4. optionally computes embedding similarity,
5. combines rankings with RRF.

This means the model sees the most relevant code chunks instead of the full repository dump.

## Testing

Run the unit tests with:

```bash
python -m unittest -v test_agent.py
```

The tests cover:

- tool contract safety,
- file access restrictions,
- BM25 retrieval,
- RRF fusion,
- README/code ranking behavior,
- prompt trimming for model context limits.

## GitHub hygiene

The repository is intentionally kept clean for GitHub:

- source files are committed,
- local environment files are excluded,
- temporary test ZIPs are excluded,
- no large local artifacts are pushed to GitHub.

This keeps the GitHub repo focused on the actual project and makes it easier for others to clone and run.

## Notes

- Semantic mode is optional and requires the dependencies in `requirements.txt`.
- The app accepts uploaded ZIPs and local files, but it is designed for safe, local-only workflows.
- Do not upload secrets or private data unless you are sure you have permission to do so.
- The evaluation ZIP used in testing is kept on your local machine and intentionally not published to GitHub.
