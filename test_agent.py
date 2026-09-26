import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import agent
import app


class ToolContractTests(unittest.TestCase):
    def test_read_file_rejects_parent_directory(self) -> None:
        self.assertEqual(
            agent.read_file("../README.md"),
            "Error: file must stay inside the project folder.",
        )

    def test_search_files_returns_path_and_line(self) -> None:
        matches = agent.search_files("def run_tool", "*.py")
        self.assertIsInstance(matches, list)
        self.assertTrue(any("agent.py:" in match for match in matches))

    def test_search_files_supports_no_match_result(self) -> None:
        query = "definitely-" + "absent-9f4e2c"
        self.assertEqual(agent.search_files(query), f"No matches for {query!r}.")

    def test_read_file_rejects_binary_content(self) -> None:
        with patch.object(Path, "read_text", side_effect=UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid")):
            self.assertEqual(agent.read_file("README.md"), "Error: file is not a UTF-8 text file.")

    def test_unknown_tool_is_safe(self) -> None:
        self.assertEqual(agent.run_tool("run_shell_command", {}), "Error: unknown tool 'run_shell_command'.")

    def test_agent_trace_records_tool_and_final_answer(self) -> None:
        trace: list[dict[str, object]] = []
        answer = agent.run_agent("Where is the tool registry?", demo=True, trace=trace)
        self.assertIn("tool returned", answer)
        self.assertEqual(trace[0]["type"], "tool_call")
        self.assertEqual(trace[0]["tool"], "search_files")
        self.assertEqual(trace[-1]["type"], "final_answer")

    def test_agent_trace_is_optional(self) -> None:
        self.assertIn("tool returned", agent.run_agent("Where is the tool registry?", demo=True))

    def test_report_contains_evidence_and_confidence(self) -> None:
        trace: list[dict[str, object]] = []
        answer = agent.run_agent("Where is the tool registry?", demo=True, trace=trace)
        report = agent.build_report("Where is the tool registry?", answer, trace)
        self.assertEqual(report["confidence"], "high")
        self.assertEqual(report["steps"], 2)
        self.assertEqual(report["evidence"][0]["tool"], "search_files")
        self.assertTrue(report["recommended_next_step"])

    def test_evidence_text_containing_error_is_not_a_tool_error(self) -> None:
        trace = [{
            "step": 1,
            "type": "tool_call",
            "tool": "search_files",
            "arguments": {"query": "Error:"},
            "status": "success",
            "result_preview": '["agent.py:1: Error: example"]',
        }]
        report = agent.build_report("Find errors", "Evidence found.", trace)
        self.assertEqual(report["confidence"], "high")

    def test_report_marks_missing_evidence_low_confidence(self) -> None:
        report = agent.build_report("Explain this", "I do not know.", [])
        self.assertEqual(report["confidence"], "low")
        self.assertEqual(report["evidence"], [])


    def test_bm25_retrieves_relevant_code(self) -> None:
        from retriever import collect_chunks, bm25_search
        chunks = collect_chunks(Path.cwd())
        results = bm25_search("reciprocal rank fusion", chunks, top_k=5)
        self.assertTrue(results)
        self.assertTrue(any("retriever.py" in item["path"] for item in results))

    def test_rrf_fuses_ranked_results(self) -> None:
        from retriever import reciprocal_rank_fusion
        a = [{"path":"a.py", "start_line":1, "source":"bm25", "text":"a"},
             {"path":"b.py", "start_line":1, "source":"bm25", "text":"b"}]
        b = [{"path":"b.py", "start_line":1, "source":"vector", "text":"b"},
             {"path":"a.py", "start_line":1, "source":"vector", "text":"a"}]
        fused = reciprocal_rank_fusion(a,b,top_k=2)
        self.assertEqual(len(fused),2)
        self.assertEqual(fused[0]["rrf_score"], fused[1]["rrf_score"])
        self.assertEqual(set(fused[0]["sources"]), {"bm25", "vector"})

    def test_hybrid_search_returns_chunk_metadata(self) -> None:
        from retriever import hybrid_search
        result = hybrid_search(Path.cwd(), "run_agent", top_k=3, semantic=False)
        self.assertIn("results", result)
        self.assertTrue(result["results"])
        self.assertIn("start_line", result["results"][0])

    def test_trim_for_model_limits_prompt_length(self) -> None:
        large_text = "A" * 20000
        trimmed = app.trim_for_model(large_text, max_chars=1200)
        self.assertLessEqual(len(trimmed), 1200)
        self.assertTrue(trimmed.startswith("A"))
        self.assertTrue(trimmed.endswith("A"))

    def test_readme_does_not_outrank_algorithm_file(self) -> None:
        from retriever import hybrid_search
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text(
                "This repository contains binary search, bubble sort, and other algorithms.\n"
                "Binary search is a common algorithm used to search sorted data.",
                encoding="utf-8",
            )
            (root / "binary_search.py").write_text(
                "def binary_search(nums, target):\n    for i, x in enumerate(nums):\n        if x == target:\n            return i\n    return -1\n",
                encoding="utf-8",
            )
            result = hybrid_search(root, "Where is the binary search implementation?", top_k=5, semantic=False)
            self.assertTrue(result["results"])
            self.assertEqual(result["results"][0]["path"], "binary_search.py")


if __name__ == "__main__":
    unittest.main()
