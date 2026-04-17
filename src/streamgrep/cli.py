from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import IO, Iterable

from streamgrep.embeddings import create_embedding_provider
from streamgrep.engine import SearchOptions, StreamingHybridSearcher
from streamgrep.types import SearchResult


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="streamgrep",
        description="Stream grep-like fulltext and semantic matches without a preprocessing step.",
    )
    parser.add_argument("query", help="Search query.")
    parser.add_argument("paths", nargs="*", default=["."], help="Files or directories to scan.")
    parser.add_argument(
        "--mode",
        choices=("hybrid", "fulltext", "semantic"),
        default="hybrid",
        help="Search mode.",
    )
    parser.add_argument(
        "--embedder",
        choices=("hashing", "sentence-transformers"),
        default="hashing",
        help="Embedding provider for semantic search.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Embedding model name for embedders that support it.",
    )
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=0.44,
        help="Minimum combined score for semantic matches.",
    )
    parser.add_argument(
        "--window-lines",
        type=int,
        default=4,
        help="Number of lines in the rolling semantic window.",
    )
    parser.add_argument(
        "--stride-lines",
        type=int,
        default=1,
        help="How many newly-read lines before scoring the next semantic window.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=320,
        help="Maximum semantic chunk size before clipping.",
    )
    parser.add_argument(
        "--max-results-per-file",
        type=int,
        default=None,
        help="Optional cap on emitted matches per file.",
    )
    parser.add_argument(
        "--hidden",
        action="store_true",
        help="Include hidden files and directories.",
    )
    output = parser.add_mutually_exclusive_group()
    output.add_argument(
        "--json",
        action="store_true",
        help="Emit newline-delimited JSON.",
    )
    output.add_argument(
        "-c",
        "--count",
        action="store_true",
        help="Only print the count of matches per file.",
    )
    output.add_argument(
        "-l",
        "--files-with-matches",
        action="store_true",
        help="Only print the names of files with matches.",
    )
    return parser


def main(argv: list[str] | None = None, *, stdout: IO[str] | None = None, stderr: IO[str] | None = None) -> int:
    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.query.strip():
        parser.error("query must not be empty")

    valid_paths = _validate_paths(args.paths, err)
    if not valid_paths:
        return 2

    try:
        provider = create_embedding_provider(args.embedder, model_name=args.model)
    except ModuleNotFoundError as exc:
        parser.error(str(exc))
    except ValueError as exc:
        parser.error(str(exc))

    searcher = StreamingHybridSearcher(provider)
    options = SearchOptions(
        mode=args.mode,
        window_lines=args.window_lines,
        stride_lines=args.stride_lines,
        max_chars=args.max_chars,
        semantic_threshold=args.semantic_threshold,
        max_results_per_file=args.max_results_per_file,
        include_hidden=args.hidden,
    )

    results = searcher.search_paths(args.query, valid_paths, options=options)
    return _write_results(results, args, out)


def _validate_paths(paths: list[str], err: IO[str]) -> list[str]:
    valid: list[str] = []
    for raw in paths:
        if Path(raw).expanduser().exists():
            valid.append(raw)
        else:
            print(f"streamgrep: {raw}: No such file or directory", file=err)
    return valid


def _write_results(results: Iterable[SearchResult], args: argparse.Namespace, out: IO[str]) -> int:
    if args.count:
        counts: Counter[str] = Counter()
        for result in results:
            counts[str(result.path)] += 1
        for path, count in counts.items():
            print(f"{path}:{count}", file=out)
        return 0 if counts else 1

    if args.files_with_matches:
        seen: set[str] = set()
        for result in results:
            key = str(result.path)
            if key not in seen:
                seen.add(key)
                print(key, file=out)
        return 0 if seen else 1

    found = False
    for result in results:
        found = True
        if args.json:
            print(json.dumps(_result_as_json(result), ensure_ascii=True), file=out)
        else:
            print(_format_result(result), file=out)

    return 0 if found else 1


def _format_result(result: SearchResult) -> str:
    location = _format_location(result.path, result.start_line, result.end_line)
    if result.kind == "fulltext":
        return f"{location}: {result.preview}"

    return (
        f"{location}: [semantic score={result.score:.3f} sim={result.similarity:.3f}] "
        f"{result.preview}"
    )


def _result_as_json(result: SearchResult) -> dict[str, object]:
    return {
        "path": str(result.path),
        "start_line": result.start_line,
        "end_line": result.end_line,
        "kind": result.kind,
        "score": round(result.score, 6),
        "similarity": round(result.similarity, 6),
        "lexical_score": round(result.lexical_score, 6),
        "matched_terms": list(result.matched_terms),
        "preview": result.preview,
    }


def _format_location(path: Path, start_line: int, end_line: int) -> str:
    rendered = str(path)
    if start_line == end_line:
        return f"{rendered}:{start_line}"
    return f"{rendered}:{start_line}-{end_line}"
