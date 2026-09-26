"""RepoLens: an evidence-first codebase triage agent."""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from retriever import hybrid_search


SYSTEM_PROMPT = """You are RepoLens, an evidence-first software engineering agent.
Investigate repository questions by searching and reading relevant files before answering.
State what the evidence shows and clearly label uncertainty. Never invent tool results.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current local date and time.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a directory below the current project folder.",
            "parameters": {
                "type": "object",
                "properties": {"directory": {"type": "string", "description": "A relative directory, usually '.'"}},
                "required": ["directory"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file below the current project folder.",
            "parameters": {
                "type": "object",
                "properties": {"file_path": {"type": "string", "description": "A relative text-file path"}},
                "required": ["file_path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search text inside project files and return matching paths and line excerpts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Case-insensitive text to find"},
                    "file_pattern": {"type": "string", "description": "Optional suffix such as '*.py'"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hybrid_code_search",
            "description": "Retrieve relevant code chunks using BM25 keyword search; optionally enable semantic embeddings and RRF fusion.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "A natural-language or code query"},
                    "top_k": {"type": "integer", "description": "Number of chunks to return", "minimum": 1, "maximum": 12},
                    "semantic": {"type": "boolean", "description": "Enable sentence-transformer embeddings and RRF"}
                },
                "required": ["query"], "additionalProperties": False
            }
        }
    },
]


def get_time() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def list_files(directory: str) -> list[str] | str:
    project_root = Path.cwd().resolve()
    requested = (project_root / directory).resolve()
    if project_root not in requested.parents and requested != project_root:
        return "Error: directory must stay inside the project folder."
    if not requested.is_dir():
        return f"Error: {directory!r} is not a directory."
    return sorted(str(path.relative_to(project_root)) for path in requested.iterdir())


def read_file(file_path: str) -> str:
    project_root = Path.cwd().resolve()
    requested = (project_root / file_path).resolve()
    if project_root not in requested.parents:
        return "Error: file must stay inside the project folder."
    if not requested.is_file():
        return f"Error: {file_path!r} is not a file."
    try:
        return requested.read_text(encoding="utf-8")[:12_000]
    except UnicodeDecodeError:
        return "Error: file is not a UTF-8 text file."


def search_files(query: str, file_pattern: str = "*") -> list[str] | str:
    project_root = Path.cwd().resolve()
    results: list[str] = []
    ignored_parts = {".git", "__pycache__"}
    for path in project_root.rglob(file_pattern):
        if not path.is_file() or ignored_parts.intersection(path.parts):
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for line_number, line in enumerate(lines, start=1):
            if query.casefold() in line.casefold():
                relative_path = path.relative_to(project_root)
                results.append(f"{relative_path}:{line_number}: {line.strip()[:240]}")
                if len(results) == 20:
                    return results
    return results or f"No matches for {query!r}."


def run_tool(name: str, arguments: dict[str, Any]) -> Any:
    """The tool registry is the agent's controlled boundary to the outside world."""
    if name == "get_time":
        return get_time()
    if name == "list_files":
        return list_files(arguments["directory"])
    if name == "read_file":
        return read_file(arguments["file_path"])
    if name == "search_files":
        return search_files(arguments["query"], arguments.get("file_pattern", "*"))
    if name == "hybrid_code_search":
        return hybrid_search(Path.cwd(), arguments["query"], int(arguments.get("top_k", 6)), bool(arguments.get("semantic", False)))
    return f"Error: unknown tool {name!r}."


def ask_model(messages: list[dict[str, Any]]) -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Run with --demo or configure an API key.")

    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    payload = json.dumps({"model": model, "messages": messages, "tools": TOOLS}).encode()
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"Model request failed ({error.code}): {detail}") from error
    return data["choices"][0]["message"]


def demo_model(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """A deterministic fake model so the architecture works before credentials do."""
    user_text = messages[1]["content"].lower()
    tool_messages = [message for message in messages if message["role"] == "tool"]
    if not tool_messages and ("where" in user_text or "search" in user_text or "find" in user_text):
        return {
            "role": "assistant",
            "content": "I will search the repository for evidence first.",
            "tool_calls": [{"id": "demo-1", "type": "function", "function": {"name": "search_files", "arguments": '{"query":"run_tool","file_pattern":"*.py"}'}}],
        }
    if not tool_messages and ("read" in user_text or "summarize" in user_text):
        return {
            "role": "assistant",
            "content": "I will read README.md before answering.",
            "tool_calls": [{"id": "demo-1", "type": "function", "function": {"name": "read_file", "arguments": '{"file_path":"README.md"}'}}],
        }
    if not tool_messages and ("file" in user_text or "folder" in user_text):
        return {
            "role": "assistant",
            "content": "I will inspect the project folder first.",
            "tool_calls": [{"id": "demo-1", "type": "function", "function": {"name": "list_files", "arguments": '{"directory":"."}'}}],
        }
    if not tool_messages and "time" in user_text:
        return {
            "role": "assistant",
            "content": "I will ask the time tool.",
            "tool_calls": [{"id": "demo-1", "type": "function", "function": {"name": "get_time", "arguments": "{}"}}],
        }
    result = tool_messages[-1]["content"] if tool_messages else "No tool was needed."
    return {"role": "assistant", "content": f"Done. The tool returned: {result}"}


def add_trace_event(trace: list[dict[str, Any]] | None, event: dict[str, Any]) -> None:
    if trace is not None:
        trace.append(event)


def run_agent(task: str, demo: bool = False, trace: list[dict[str, Any]] | None = None) -> str:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    model_call = demo_model if demo else ask_model

    for step in range(1, 6):
        response = model_call(messages)
        messages.append(response)
        tool_calls = response.get("tool_calls", [])
        if not tool_calls:
            add_trace_event(
                trace,
                {"step": step, "type": "final_answer", "content": response.get("content", "")},
            )
            return response.get("content", "")
        for tool_call in tool_calls:
            function = tool_call["function"]
            arguments = json.loads(function.get("arguments", "{}"))
            result = run_tool(function["name"], arguments)
            add_trace_event(
                trace,
                {
                    "step": step,
                    "type": "tool_call",
                    "tool": function["name"],
                    "arguments": arguments,
                    "status": "error" if isinstance(result, str) and result.startswith("Error:") else "success",
                    "result_preview": json.dumps(result)[:1_000],
                },
            )
            messages.append({"role": "tool", "tool_call_id": tool_call["id"], "content": json.dumps(result)})
    raise RuntimeError("The agent exceeded its five-step limit.")


def build_report(task: str, answer: str, trace: list[dict[str, Any]]) -> dict[str, Any]:
    tool_events = [event for event in trace if event.get("type") == "tool_call"]
    evidence = [
        {
            "tool": event["tool"],
            "arguments": event["arguments"],
            "excerpt": event["result_preview"],
        }
        for event in tool_events
    ]
    has_error = any(event.get("status") == "error" for event in tool_events)
    if not evidence:
        confidence = "low"
        next_step = "Collect repository evidence before relying on this answer."
    elif has_error:
        confidence = "low"
        next_step = "Resolve the tool error and repeat the investigation."
    else:
        confidence = "high"
        next_step = "Review the cited evidence and verify the conclusion in the relevant files."
    return {
        "task": task,
        "findings": [answer],
        "evidence": evidence,
        "confidence": confidence,
        "recommended_next_step": next_step,
        "steps": len(trace),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny tool-using AI agent.")
    parser.add_argument("task", help="The task for the agent")
    parser.add_argument("--demo", action="store_true", help="Use a fake model; no API key required")
    parser.add_argument("--trace", action="store_true", help="Print the investigation trace after the answer")
    parser.add_argument("--report", action="store_true", help="Print a structured JSON investigation report")
    args = parser.parse_args()
    trace: list[dict[str, Any]] = []
    answer = run_agent(args.task, demo=args.demo, trace=trace)
    if args.report:
        print(json.dumps(build_report(args.task, answer, trace), indent=2))
        return
    print(answer)
    if args.trace:
        print("\nInvestigation trace:")
        print(json.dumps(trace, indent=2))


if __name__ == "__main__":
    main()
