from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class RollingChunk:
    start_line: int
    end_line: int
    text: str


class RollingChunker:
    def __init__(self, *, window_lines: int, stride_lines: int, max_chars: int) -> None:
        if window_lines < 1:
            raise ValueError("window_lines must be >= 1")
        if stride_lines < 1:
            raise ValueError("stride_lines must be >= 1")
        if max_chars < 32:
            raise ValueError("max_chars must be >= 32")

        self.window_lines = window_lines
        self.stride_lines = stride_lines
        self.max_chars = max_chars
        self._buffer: deque[tuple[int, str]] = deque()
        self._lines_since_emit = 0

    def push(self, line_number: int, text: str) -> RollingChunk | None:
        self._buffer.append((line_number, text.rstrip("\n")))
        while len(self._buffer) > self.window_lines:
            self._buffer.popleft()

        self._lines_since_emit += 1
        if self._lines_since_emit < self.stride_lines:
            return None

        self._lines_since_emit = 0
        return self._build_chunk()

    def flush(self) -> RollingChunk | None:
        if not self._buffer or self._lines_since_emit == 0:
            return None
        self._lines_since_emit = 0
        return self._build_chunk()

    def _build_chunk(self) -> RollingChunk | None:
        parts = [line.strip() for _, line in self._buffer if line.strip()]
        if not parts:
            return None

        text = re.sub(r"\s+", " ", " ".join(parts)).strip()
        if len(text) > self.max_chars:
            clipped = text[: self.max_chars].rstrip()
            if " " in clipped:
                clipped = clipped.rsplit(" ", 1)[0]
            text = clipped

        return RollingChunk(
            start_line=self._buffer[0][0],
            end_line=self._buffer[-1][0],
            text=text,
        )
