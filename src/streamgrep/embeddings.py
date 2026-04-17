from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from typing import Protocol

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")

_SYNONYM_GROUPS = {
    "auth": {
        "auth",
        "authenticate",
        "authenticated",
        "authentication",
        "authorisation",
        "authorization",
        "credential",
        "credentials",
        "login",
        "logins",
        "password",
        "passwords",
        "signin",
        "signon",
    },
    "error": {
        "bug",
        "bugs",
        "crash",
        "crashes",
        "error",
        "errors",
        "exception",
        "exceptions",
        "fail",
        "failed",
        "failing",
        "failure",
        "failures",
        "issue",
        "issues",
    },
    "search": {
        "discover",
        "find",
        "lookup",
        "query",
        "search",
    },
    "semantic": {
        "embedding",
        "embeddings",
        "meaning",
        "semantic",
        "semantics",
        "similarity",
        "vector",
        "vectors",
    },
    "stream": {
        "incremental",
        "live",
        "realtime",
        "stream",
        "streaming",
    },
}

_SYNONYM_LOOKUP = {
    variant: root
    for root, variants in _SYNONYM_GROUPS.items()
    for variant in variants
}


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


def normalize_terms(text: str) -> list[str]:
    normalized: list[str] = []
    for token in TOKEN_RE.findall(text.lower()):
        collapsed = token.replace("-", "").replace("_", "").replace(".", "")
        stemmed = _simple_stem(collapsed)
        canonical = _SYNONYM_LOOKUP.get(stemmed, stemmed)
        if canonical:
            normalized.append(canonical)
    return normalized


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    numerator = sum(lhs * rhs for lhs, rhs in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


@dataclass
class HashingEmbeddingProvider:
    dimensions: int = 512

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        terms = normalize_terms(text)

        for term in terms:
            self._accumulate(vector, f"tok:{term}", 1.0)
            for trigram in _char_ngrams(term, 3):
                self._accumulate(vector, f"tri:{trigram}", 0.14)

        for index in range(len(terms) - 1):
            bigram = f"{terms[index]}_{terms[index + 1]}"
            self._accumulate(vector, f"bi:{bigram}", 0.35)

        return _l2_normalize(vector)

    def _accumulate(self, vector: list[float], feature: str, weight: float) -> None:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % self.dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign * weight


class SentenceTransformerEmbeddingProvider:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        import importlib.util

        if importlib.util.find_spec("sentence_transformers") is None:
            raise ModuleNotFoundError(
                "sentence-transformers is not installed; "
                "install optional dependencies with `pip install -e .[semantic]`"
            )
        self.model_name = model_name
        self._model: object | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._load_model()
        encoded = model.encode(texts, normalize_embeddings=True)
        return [list(map(float, row)) for row in encoded]

    def _load_model(self) -> object:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model


def create_embedding_provider(name: str, *, model_name: str | None = None) -> EmbeddingProvider:
    if name == "hashing":
        return HashingEmbeddingProvider()
    if name == "sentence-transformers":
        return SentenceTransformerEmbeddingProvider(
            model_name=model_name or "sentence-transformers/all-MiniLM-L6-v2"
        )
    raise ValueError(f"Unsupported embedder: {name}")


def _simple_stem(token: str) -> str:
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _char_ngrams(token: str, size: int) -> list[str]:
    if len(token) < size:
        return [token]
    return [token[index : index + size] for index in range(len(token) - size + 1)]


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]
