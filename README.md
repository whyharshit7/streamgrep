# streamgrep

`streamgrep` is a grep-like prototype for hybrid search:

- No indexing or pre-processing pass.
- Fulltext hits stream immediately, like `grep`.
- Semantic hits also stream immediately as each file is read and chunked.
- Embeddings are pluggable, so you can start with a zero-dependency baseline and swap in a real model later.

## Why this shape works

Traditional vector search systems usually want a full corpus ingest before they can answer anything. `streamgrep` flips that around:

1. Compute the query embedding once.
2. Walk files directly from disk.
3. Emit exact/fulltext matches line-by-line.
4. Maintain a rolling semantic window while reading each file.
5. Embed each window on the fly and emit semantic hits as soon as they clear a threshold.

That gives you grep-style responsiveness with a semantic lane layered on top.

## Quick start

Run the CLI directly from the workspace:

```bash
PYTHONPATH=src python3 -m streamgrep "login error" .
```

Example output:

```text
src/app/auth.py:18: return {"error": "invalid password"}
src/app/auth.py:14-18: [semantic score=0.731 sim=0.681] the signin flow rejects valid credentials after a token refresh
```

## Modes

- `--mode hybrid`: fulltext + semantic together.
- `--mode fulltext`: grep-like exact/token matching only.
- `--mode semantic`: semantic-only rolling windows.

## Embedders

Default:

- `--embedder hashing`
- No extra dependencies.
- Uses feature hashing + token normalization + lightweight synonym collapsing.
- Good enough for a local prototype and tests.

Optional:

- `--embedder sentence-transformers`
- Install with `pip install -e .[semantic]`
- Pass a model with `--model sentence-transformers/all-MiniLM-L6-v2`

## Useful flags

```bash
PYTHONPATH=src python3 -m streamgrep "payment failure" src tests \
  --mode hybrid \
  --window-lines 4 \
  --stride-lines 1 \
  --semantic-threshold 0.44
```

- `--window-lines`: size of the rolling semantic context.
- `--stride-lines`: how often to score a new rolling chunk.
- `--max-chars`: cap chunk size before embedding.
- `--json`: emit newline-delimited JSON instead of human-readable text.
- `--hidden`: include hidden files and directories.
- `-j/--workers`: search multiple files concurrently. Per-file streaming order
  is preserved, and global output follows the file-iteration order regardless
  of which worker finishes first.

## Design notes

- The current implementation favors streaming latency over perfect ranking.
- Semantic deduping is intentionally light so nearby windows do not spam the console.
- The searcher skips likely binary files and common large dependency directories.
- There is no offline corpus build step, which keeps the operational model close to `grep`.

## Next improvements

- Add model-specific batching for semantic windows to improve throughput.
- Add incremental reranking so the best-so-far results can be surfaced while the scan is still in progress.
- Add `--follow` and filesystem watch support for long-running live search sessions.
