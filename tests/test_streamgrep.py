from __future__ import annotations

from pathlib import Path
import io
import tempfile
import unittest
from unittest.mock import patch

from streamgrep.chunking import RollingChunker
from streamgrep.cli import main
from streamgrep.embeddings import (
    HashingEmbeddingProvider,
    cosine_similarity,
    create_embedding_provider,
)
from streamgrep.engine import SearchOptions, StreamingHybridSearcher
from streamgrep.files import iter_searchable_files
from streamgrep.types import SearchResult


class TrackedLines:
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines
        self.read_count = 0

    def __iter__(self):
        for line in self.lines:
            self.read_count += 1
            yield line


class StreamGrepTests(unittest.TestCase):
    def test_hashing_embedder_prefers_related_text(self) -> None:
        provider = HashingEmbeddingProvider()
        query_vector = provider.embed(["login failure"])[0]
        related_vector = provider.embed(["authentication error on sign in"])[0]
        unrelated_vector = provider.embed(["sunlight on a quiet beach at dusk"])[0]

        related_score = cosine_similarity(query_vector, related_vector)
        unrelated_score = cosine_similarity(query_vector, unrelated_vector)

        self.assertGreater(related_score, unrelated_score)

    def test_semantic_result_streams_before_the_file_is_fully_read(self) -> None:
        provider = HashingEmbeddingProvider()
        searcher = StreamingHybridSearcher(provider)
        prepared = searcher.prepare_query("login failure", include_semantic=True)
        lines = TrackedLines(
            [
                "The authentication error starts here.\n",
                "More unrelated details follow later.\n",
                "Still scanning the rest of the file.\n",
            ]
        )

        generator = searcher.search_lines(
            Path("demo.txt"),
            lines,
            prepared,
            options=SearchOptions(mode="semantic", window_lines=1, stride_lines=1),
        )

        first_result = next(generator)

        self.assertEqual(first_result.kind, "semantic")
        self.assertEqual(first_result.start_line, 1)
        self.assertEqual(lines.read_count, 1)

    def test_search_paths_finds_fulltext_and_semantic_hits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            file_path = root / "notes.txt"
            file_path.write_text(
                "\n".join(
                    [
                        "A plain introduction.",
                        "The login system throws an error after a token refresh.",
                        "Footer text.",
                    ]
                ),
                encoding="utf-8",
            )

            provider = HashingEmbeddingProvider()
            searcher = StreamingHybridSearcher(provider)
            results = list(
                searcher.search_paths(
                    "login error",
                    [str(root)],
                    options=SearchOptions(mode="hybrid", window_lines=2, stride_lines=1),
                )
            )

            result_kinds = {result.kind for result in results}

            self.assertIn("semantic", result_kinds)
            self.assertIn("fulltext", result_kinds)

    def test_semantic_threshold_filters_results(self) -> None:
        provider = HashingEmbeddingProvider()
        searcher = StreamingHybridSearcher(provider)
        prepared = searcher.prepare_query("login failure", include_semantic=True)
        results = list(
            searcher.search_lines(
                Path("demo.txt"),
                ["The authentication error starts here.\n"],
                prepared,
                options=SearchOptions(
                    mode="semantic",
                    window_lines=1,
                    stride_lines=1,
                    semantic_threshold=0.95,
                ),
            )
        )

        self.assertEqual(results, [])

    def test_file_discovery_is_lazy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first_file = root / "first.txt"
            first_file.write_text("hello\n", encoding="utf-8")

            def fake_walk(_path: Path):
                yield (str(root), [], ["first.txt"])
                raise AssertionError("walked past the first yield")

            with patch("streamgrep.files.os.walk", fake_walk):
                generator = iter_searchable_files([str(root)])
                discovered = next(generator)

            self.assertEqual(discovered, first_file)

    def test_binary_files_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "text.txt").write_text("plain text\n", encoding="utf-8")
            (root / "blob.bin").write_bytes(b"prefix\x00binary")

            discovered = sorted(path.name for path in iter_searchable_files([str(root)]))

            self.assertEqual(discovered, ["text.txt"])

    def test_rolling_chunker_flush_emits_tail(self) -> None:
        chunker = RollingChunker(window_lines=3, stride_lines=5, max_chars=64)
        chunker.push(1, "alpha")
        chunker.push(2, "beta")
        tail = chunker.flush()

        self.assertIsNotNone(tail)
        assert tail is not None
        self.assertEqual(tail.start_line, 1)
        self.assertEqual(tail.end_line, 2)
        self.assertIn("alpha", tail.text)
        self.assertIn("beta", tail.text)

    def test_rolling_chunker_rejects_invalid_params(self) -> None:
        with self.assertRaises(ValueError):
            RollingChunker(window_lines=0, stride_lines=1, max_chars=64)
        with self.assertRaises(ValueError):
            RollingChunker(window_lines=1, stride_lines=0, max_chars=64)
        with self.assertRaises(ValueError):
            RollingChunker(window_lines=1, stride_lines=1, max_chars=16)

    def test_semantic_dedup_suppresses_near_duplicates(self) -> None:
        searcher = StreamingHybridSearcher(HashingEmbeddingProvider())
        prior = SearchResult(
            path=Path("demo.txt"),
            start_line=10,
            end_line=14,
            kind="semantic",
            score=0.80,
            similarity=0.80,
            lexical_score=0.0,
            preview="",
        )
        near_duplicate = SearchResult(
            path=Path("demo.txt"),
            start_line=11,
            end_line=15,
            kind="semantic",
            score=0.81,
            similarity=0.81,
            lexical_score=0.0,
            preview="",
        )
        distinct = SearchResult(
            path=Path("demo.txt"),
            start_line=100,
            end_line=104,
            kind="semantic",
            score=0.80,
            similarity=0.80,
            lexical_score=0.0,
            preview="",
        )

        self.assertFalse(searcher._is_novel_semantic(near_duplicate, [prior]))
        self.assertTrue(searcher._is_novel_semantic(distinct, [prior]))

    def test_cli_count_mode_prints_totals_per_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "a.txt").write_text("login here\nlogin again\nunrelated\n", encoding="utf-8")
            (root / "b.txt").write_text("nothing to see\n", encoding="utf-8")

            stdout = io.StringIO()
            stderr = io.StringIO()
            exit_code = main(
                ["login", str(root), "--mode", "fulltext", "--count"],
                stdout=stdout,
                stderr=stderr,
            )

            self.assertEqual(exit_code, 0)
            lines = [line for line in stdout.getvalue().splitlines() if line]
            self.assertEqual(len(lines), 1)
            self.assertTrue(lines[0].endswith(":2"))

    def test_cli_files_with_matches_lists_each_file_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "a.txt").write_text("login here\nlogin again\n", encoding="utf-8")
            (root / "b.txt").write_text("login somewhere else\n", encoding="utf-8")

            stdout = io.StringIO()
            stderr = io.StringIO()
            exit_code = main(
                ["login", str(root), "--mode", "fulltext", "-l"],
                stdout=stdout,
                stderr=stderr,
            )

            self.assertEqual(exit_code, 0)
            lines = [line for line in stdout.getvalue().splitlines() if line]
            self.assertEqual(len(lines), 2)
            self.assertEqual(len(set(lines)), 2)

    def test_cli_warns_on_missing_path(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        exit_code = main(
            ["foo", "/definitely/not/a/real/path/streamgrep"],
            stdout=stdout,
            stderr=stderr,
        )

        self.assertEqual(exit_code, 2)
        self.assertIn("No such file or directory", stderr.getvalue())

    def test_cli_rejects_empty_query(self) -> None:
        with patch("sys.stderr", new=io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                main(["   "], stdout=io.StringIO(), stderr=io.StringIO())
        self.assertEqual(ctx.exception.code, 2)

    def test_sentence_transformers_raises_on_missing_dependency(self) -> None:
        with patch("importlib.util.find_spec", return_value=None):
            with self.assertRaises(ModuleNotFoundError):
                create_embedding_provider("sentence-transformers")


if __name__ == "__main__":
    unittest.main()
