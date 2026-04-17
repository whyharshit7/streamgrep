from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Iterator

from streamgrep.chunking import RollingChunk, RollingChunker
from streamgrep.embeddings import EmbeddingProvider, cosine_similarity, normalize_terms
from streamgrep.files import iter_searchable_files
from streamgrep.types import SearchResult

WHITESPACE_RE = re.compile(r"\s+")
_SEMANTIC_DEDUP_WINDOW = 4


@dataclass(frozen=True)
class SearchOptions:
    mode: str = "hybrid"
    window_lines: int = 4
    stride_lines: int = 1
    max_chars: int = 320
    semantic_threshold: float = 0.44
    max_results_per_file: int | None = None
    include_hidden: bool = False


@dataclass(frozen=True)
class PreparedQuery:
    raw: str
    lowered: str
    terms: tuple[str, ...]
    vector: list[float] | None


class StreamingHybridSearcher:
    def __init__(self, embedding_provider: EmbeddingProvider) -> None:
        self.embedding_provider = embedding_provider

    def search_paths(
        self,
        query: str,
        paths: list[str],
        *,
        options: SearchOptions,
    ) -> Iterator[SearchResult]:
        prepared = self.prepare_query(query, include_semantic=options.mode != "fulltext")
        for path in iter_searchable_files(paths, include_hidden=options.include_hidden):
            yield from self.search_file(path, prepared, options=options)

    def prepare_query(self, query: str, *, include_semantic: bool) -> PreparedQuery:
        lowered = query.strip().lower()
        terms = tuple(dict.fromkeys(normalize_terms(lowered)))
        vector = None
        if include_semantic:
            vector = self.embedding_provider.embed([query])[0]
        return PreparedQuery(raw=query, lowered=lowered, terms=terms, vector=vector)

    def search_file(
        self,
        path: Path,
        prepared: PreparedQuery,
        *,
        options: SearchOptions,
    ) -> Iterator[SearchResult]:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            yield from self.search_lines(path, handle, prepared, options=options)

    def search_lines(
        self,
        path: Path,
        lines: Iterable[str],
        prepared: PreparedQuery,
        *,
        options: SearchOptions,
    ) -> Iterator[SearchResult]:
        chunker = RollingChunker(
            window_lines=options.window_lines,
            stride_lines=options.stride_lines,
            max_chars=options.max_chars,
        )
        emitted_semantic: deque[SearchResult] = deque(maxlen=_SEMANTIC_DEDUP_WINDOW)
        emitted_count = 0

        for line_number, raw_line in enumerate(lines, start=1):
            line = raw_line.rstrip("\n")

            if options.mode in {"hybrid", "fulltext"}:
                fulltext_result = self._fulltext_result(path, line_number, line, prepared)
                if fulltext_result is not None:
                    yield fulltext_result
                    emitted_count += 1
                    if self._hit_limit(options, emitted_count):
                        return

            if options.mode in {"hybrid", "semantic"}:
                chunk = chunker.push(line_number, line)
                semantic_result = self._semantic_result(
                    path,
                    chunk,
                    prepared,
                    semantic_threshold=options.semantic_threshold,
                )
                if semantic_result is not None and self._is_novel_semantic(
                    semantic_result, emitted_semantic
                ):
                    emitted_semantic.append(semantic_result)
                    yield semantic_result
                    emitted_count += 1
                    if self._hit_limit(options, emitted_count):
                        return

        if options.mode in {"hybrid", "semantic"}:
            tail_chunk = chunker.flush()
            tail_result = self._semantic_result(
                path,
                tail_chunk,
                prepared,
                semantic_threshold=options.semantic_threshold,
            )
            if tail_result is not None and self._is_novel_semantic(tail_result, emitted_semantic):
                yield tail_result
                emitted_count += 1
                if self._hit_limit(options, emitted_count):
                    return

    def _fulltext_result(
        self,
        path: Path,
        line_number: int,
        line: str,
        prepared: PreparedQuery,
    ) -> SearchResult | None:
        lexical_score, matched_terms, phrase_match = self._lexical_score(line, prepared)
        has_hit = phrase_match or (
            prepared.terms and len(prepared.terms) == 1 and lexical_score > 0.0
        ) or (prepared.terms and matched_terms == prepared.terms)

        if not has_hit:
            return None

        return SearchResult(
            path=path,
            start_line=line_number,
            end_line=line_number,
            kind="fulltext",
            score=1.0 if phrase_match else lexical_score,
            similarity=0.0,
            lexical_score=lexical_score,
            preview=_preview_text(line),
            matched_terms=matched_terms,
        )

    def _semantic_result(
        self,
        path: Path,
        chunk: RollingChunk | None,
        prepared: PreparedQuery,
        *,
        semantic_threshold: float,
    ) -> SearchResult | None:
        if chunk is None or prepared.vector is None:
            return None

        lexical_score, matched_terms, phrase_match = self._lexical_score(chunk.text, prepared)
        candidate_vector = self.embedding_provider.embed([chunk.text])[0]
        similarity = cosine_similarity(prepared.vector, candidate_vector)
        combined_score = max(similarity, (0.72 * similarity) + (0.28 * lexical_score))
        if phrase_match:
            combined_score = min(1.0, combined_score + 0.08)
        if combined_score < semantic_threshold:
            return None

        return SearchResult(
            path=path,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            kind="semantic",
            score=combined_score,
            similarity=similarity,
            lexical_score=lexical_score,
            preview=_preview_text(chunk.text),
            matched_terms=matched_terms,
        )

    def _lexical_score(
        self,
        text: str,
        prepared: PreparedQuery,
    ) -> tuple[float, tuple[str, ...], bool]:
        phrase_match = prepared.lowered in text.lower()
        if not prepared.terms:
            return (1.0 if phrase_match else 0.0, tuple(), phrase_match)

        present_terms = set(normalize_terms(text))
        matched_terms = tuple(term for term in prepared.terms if term in present_terms)
        lexical_score = len(matched_terms) / len(prepared.terms)
        if phrase_match:
            lexical_score = min(1.0, lexical_score + 0.25)
        return lexical_score, matched_terms, phrase_match

    def _is_novel_semantic(
        self,
        candidate: SearchResult,
        previous: Iterable[SearchResult],
    ) -> bool:
        for prior in previous:
            if prior.path != candidate.path:
                continue
            overlap_start = max(prior.start_line, candidate.start_line)
            overlap_end = min(prior.end_line, candidate.end_line)
            if overlap_end < overlap_start:
                continue
            overlap = overlap_end - overlap_start + 1
            largest_span = max(
                prior.end_line - prior.start_line + 1,
                candidate.end_line - candidate.start_line + 1,
            )
            if overlap / largest_span >= 0.75 and abs(prior.score - candidate.score) <= 0.04:
                return False
        return True

    def _hit_limit(self, options: SearchOptions, emitted_count: int) -> bool:
        return options.max_results_per_file is not None and emitted_count >= options.max_results_per_file


def _preview_text(text: str, *, width: int = 140) -> str:
    compact = WHITESPACE_RE.sub(" ", text).strip()
    if len(compact) <= width:
        return compact
    clipped = compact[:width].rstrip()
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return f"{clipped}..."
