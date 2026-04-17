from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

SearchKind = Literal["fulltext", "semantic"]


@dataclass(frozen=True)
class SearchResult:
    path: Path
    start_line: int
    end_line: int
    kind: SearchKind
    score: float
    similarity: float
    lexical_score: float
    preview: str
    matched_terms: tuple[str, ...] = field(default_factory=tuple)
