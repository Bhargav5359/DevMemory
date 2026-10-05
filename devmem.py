#!/usr/bin/env python3
"""
devmem.py — Zero-cost, local RAG CLI for developer context memory.

Indexes shell history, VS Code workspaces, and online platform solutions
(e.g. LeetCode) via ChromaDB + Ollama — completely offline.
"""

import hashlib
import os
import re
import sys
from pathlib import Path

import chromadb
import click
import ollama

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

DB_PATH = Path.home() / ".devmem_db"
COLLECTION_NAME = "dev_memory"

SUPPORTED_EXTENSIONS = {".py", ".ts", ".js", ".go", ".rs", ".cpp", ".java", ".md"}
IGNORED_DIRS = {"node_modules", ".git", "venv", "__pycache__", "dist", "build"}

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2"

CHUNK_SIZE = 50  # lines per code chunk

# ──────────────────────────────────────────────
# ChromaDB Client & Collection
# ──────────────────────────────────────────────

def get_collection() -> chromadb.Collection:
    """Return a persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=str(DB_PATH))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# ──────────────────────────────────────────────
# Embedding Helper
# ──────────────────────────────────────────────

def get_embedding(text: str) -> list[float]:
    """Generate an embedding vector for *text* using Ollama nomic-embed-text."""
    try:
        response = ollama.embed(model=EMBED_MODEL, input=text)
        # ollama SDK >=0.2: response.embeddings is a list of vectors
        embeddings = response.get("embeddings") or response.get("embedding")
        if embeddings is None:
            raise ValueError("Ollama returned no embeddings field.")
        # embed() with a single string returns a list with one vector
        if isinstance(embeddings[0], list):
            return embeddings[0]
        return embeddings
    except Exception as exc:
        click.echo(f"[ERROR] Embedding failed: {exc}", err=True)
        sys.exit(1)


# ──────────────────────────────────────────────
# Deduplication Helper
# ──────────────────────────────────────────────

def make_doc_id(text: str) -> str:
    """Return a stable, hashed document ID for *text*."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"doc_{digest}"


# ──────────────────────────────────────────────
# Batch Upsert Helper
# ──────────────────────────────────────────────

def upsert_documents(
    collection: chromadb.Collection,
    documents: list[str],
    metadatas: list[dict],
) -> tuple[int, int]:
    """
    Embed and upsert *documents* into *collection*.

    Returns (added, skipped) counts.
    """
    if not documents:
        return 0, 0

    existing_ids = set(collection.get(include=[])["ids"])

    to_add_docs, to_add_ids, to_add_metas = [], [], []
    skipped = 0

    for doc, meta in zip(documents, metadatas):
        doc_id = make_doc_id(doc)
        if doc_id in existing_ids:
            skipped += 1
        else:
            to_add_docs.append(doc)
            to_add_ids.append(doc_id)
            to_add_metas.append(meta)

    if not to_add_docs:
        return 0, skipped

    # Embed documents one-by-one with progress feedback
    embeddings = []
    for i, doc in enumerate(to_add_docs):
        click.echo(f"  Embedding {i + 1}/{len(to_add_docs)}...", nl=False)
        emb = get_embedding(doc)
        embeddings.append(emb)
        click.echo(" done")

    collection.upsert(
        ids=to_add_ids,
        documents=to_add_docs,
        embeddings=embeddings,
        metadatas=to_add_metas,
    )
    return len(to_add_docs), skipped


# ──────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────

@click.group()
def cli():
    """devmem -- your local developer memory, powered by RAG."""


# ──────────────────────────────────────────────
# Command: sync
# ──────────────────────────────────────────────

@cli.command("sync")
@click.option(
    "--limit",
    default=500,
    show_default=True,
    help="Maximum number of recent shell commands to index.",
)
def sync(limit: int):
    """Index recent shell history (zsh or bash)."""

    # Locate history file
    history_file = Path.home() / ".zsh_history"
    if not history_file.exists():
        history_file = Path.home() / ".bash_history"
    if not history_file.exists():
        click.echo(
            "[ERROR] No shell history file found (~/.zsh_history or ~/.bash_history).",
            err=True,
        )
        sys.exit(1)

    click.echo(f"Reading shell history from: {history_file}")

    raw_lines = history_file.read_text(encoding="utf-8", errors="replace").splitlines()

    # Clean ZSH extended history format: `: <timestamp>:<elapsed>;<command>`
    zsh_prefix_re = re.compile(r"^:\s*\d+:\d+;")
    cleaned: list[str] = []
    for line in raw_lines:
        line = zsh_prefix_re.sub("", line).strip()
        # Skip blank, too-short, or recursive devmem calls
        if len(line) < 5:
            continue
        if "devmem" in line:
            continue
        cleaned.append(line)

    # Deduplicate while preserving recency order (keep last occurrence index)
    seen: dict[str, int] = {}
    for idx, cmd in enumerate(cleaned):
        seen[cmd] = idx
    unique_cmds = [cmd for cmd, _ in sorted(seen.items(), key=lambda x: x[1])]

    # Apply limit (most recent)
    unique_cmds = unique_cmds[-limit:]
    click.echo(f"Found {len(unique_cmds)} unique commands to index.")

    collection = get_collection()
    documents = unique_cmds
    metadatas = [{"source": "shell_command"} for _ in documents]

    added, skipped = upsert_documents(collection, documents, metadatas)
    click.echo(f"\nShell sync complete -- {added} added, {skipped} skipped (duplicates).")


# ──────────────────────────────────────────────
# Command: sync-code
# ──────────────────────────────────────────────

@cli.command("sync-code")
@click.argument(
    "workspace_dir",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
)
def sync_code(workspace_dir: str):
    """Index source code files from WORKSPACE_DIR into ChromaDB."""

    root = Path(workspace_dir)
    click.echo(f"Scanning workspace: {root}")

    documents: list[str] = []
    metadatas: list[dict] = []

    for file_path in root.rglob("*"):
        # Skip ignored directories anywhere in the path
        if any(part in IGNORED_DIRS for part in file_path.parts):
            continue
        if file_path.suffix not in SUPPORTED_EXTENSIONS:
            continue
        if not file_path.is_file():
            continue

        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            click.echo(f"  [WARN] Could not read {file_path}: {exc}", err=True)
            continue

        lines = content.splitlines()
        if not lines:
            continue

        # Chunk into CHUNK_SIZE-line blocks
        for start in range(0, len(lines), CHUNK_SIZE):
            chunk_lines = lines[start : start + CHUNK_SIZE]
            chunk_text = "\n".join(chunk_lines).strip()
            if not chunk_text:
                continue

            documents.append(chunk_text)
            metadatas.append(
                {
                    "source": "vscode",
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "start_line": start + 1,
                }
            )

    click.echo(f"Found {len(documents)} code chunks to index.")

    if not documents:
        click.echo("No code chunks found. Check your workspace path and file extensions.")
        return

    collection = get_collection()
    added, skipped = upsert_documents(collection, documents, metadatas)
    click.echo(f"\nCode sync complete -- {added} added, {skipped} skipped (duplicates).")


# ──────────────────────────────────────────────
# Command: sync-leetcode
# ──────────────────────────────────────────────

@cli.command("sync-leetcode")
@click.argument(
    "repo_path",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
)
def sync_leetcode(repo_path: str):
    """Index local LeetCode / HackerRank solution files from REPO_PATH."""

    root = Path(repo_path)
    click.echo(f"Scanning online-platform solutions in: {root}")

    documents: list[str] = []
    metadatas: list[dict] = []

    for file_path in root.rglob("*"):
        if any(part in IGNORED_DIRS for part in file_path.parts):
            continue
        if not file_path.is_file():
            continue
        # Accept solution files (code + text + SQL)
        accepted_suffixes = SUPPORTED_EXTENSIONS | {".txt", ".sql", ".sh"}
        if file_path.suffix not in accepted_suffixes:
            continue

        try:
            content = file_path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError as exc:
            click.echo(f"  [WARN] Could not read {file_path}: {exc}", err=True)
            continue

        if not content:
            continue

        # Use the parent folder name as the problem name (common pattern for
        # auto-sync repos where each problem lives in its own directory).
        problem_name = (
            file_path.parent.name if file_path.parent != root else file_path.stem
        )

        documents.append(content)
        metadatas.append(
            {
                "source": "online_platform",
                "problem": problem_name,
                "file_path": str(file_path),
            }
        )

    click.echo(f"Found {len(documents)} solution files to index.")

    if not documents:
        click.echo("No solution files found. Check your repo path.")
        return

    collection = get_collection()
    added, skipped = upsert_documents(collection, documents, metadatas)
    click.echo(f"\nLeetCode sync complete -- {added} added, {skipped} skipped (duplicates).")


# ──────────────────────────────────────────────
# Command: ask
# ──────────────────────────────────────────────

def _format_context_block(doc: str, meta: dict) -> str:
    """Return a labelled context block string for a retrieved document."""
    source = meta.get("source", "unknown")

    if source == "shell_command":
        label = "[Terminal Command]"
    elif source == "vscode":
        fname = meta.get("file_name", "unknown")
        start = meta.get("start_line", "?")
        label = f"[VS Code | {fname} (Line {start})]"
    elif source == "online_platform":
        problem = meta.get("problem", "unknown")
        label = f"[LeetCode/Platform | {problem}]"
    else:
        label = f"[{source}]"

    return f"{label}\n{doc}"


@cli.command("ask")
@click.argument("query")
@click.option(
    "--top-k",
    default=5,
    show_default=True,
    help="Number of top context matches to retrieve.",
)
def ask(query: str, top_k: int):
    """Ask a question against your indexed developer memory."""

    collection = get_collection()

    total = collection.count()
    if total == 0:
        click.echo(
            "[WARN] The database is empty. Run `devmem sync` / `devmem sync-code` first."
        )
        return

    click.echo(f'Querying memory for: "{query}"')

    # Embed the query
    query_embedding = get_embedding(query)

    # Retrieve top-K results
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, total),
        include=["documents", "metadatas"],
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    if not docs:
        click.echo(
            "No relevant context found. Try syncing more data or rephrasing your query."
        )
        return

    # Build structured context string
    context_blocks = [_format_context_block(d, m) for d, m in zip(docs, metas)]
    context = "\n\n---\n\n".join(context_blocks)

    system_prompt = (
        "You are DevMem, an expert developer assistant with access to the user's "
        "personal developer context (shell commands, code snippets, and platform solutions). "
        "Answer concisely: provide the exact command or code solution, followed by a "
        "1-sentence explanation, and cite the source label (e.g., [Terminal Command], "
        "[VS Code | filename (Line X)], or [LeetCode/Platform | problem]) where relevant."
    )

    user_message = (
        f"Using the following context from my developer memory, answer this question:\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{context}"
    )

    click.echo("\nGenerating answer...\n")

    try:
        response = ollama.chat(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        answer = response["message"]["content"]
    except Exception as exc:
        click.echo(f"[ERROR] Ollama chat failed: {exc}", err=True)
        click.echo(
            "Make sure Ollama is running (`ollama serve`) and llama3.2 is pulled.",
            err=True,
        )
        sys.exit(1)

    click.echo("-" * 60)
    click.echo(answer)
    click.echo("-" * 60)


# ──────────────────────────────────────────────
# Command: stats
# ──────────────────────────────────────────────

@cli.command("stats")
def stats():
    """Print indexed item counts grouped by source type."""

    collection = get_collection()
    total = collection.count()

    if total == 0:
        click.echo("Database is empty. Run a sync command first.")
        return

    # Fetch all metadatas (no embeddings/documents needed)
    all_data = collection.get(include=["metadatas"])
    all_metas = all_data.get("metadatas", [])

    counts: dict[str, int] = {}
    for meta in all_metas:
        src = meta.get("source", "unknown") if meta else "unknown"
        counts[src] = counts.get(src, 0) + 1

    SOURCE_LABELS = {
        "shell_command":     "Shell Commands",
        "vscode":            "VS Code / Source Files",
        "online_platform":   "Online Platform (LeetCode etc.)",
    }

    click.echo("\nDevMem -- Indexed Item Statistics")
    click.echo("-" * 40)
    for src, count in sorted(counts.items(), key=lambda x: -x[1]):
        label = SOURCE_LABELS.get(src, src)
        click.echo(f"  {label:<36}  {count:>6}")
    click.echo("-" * 40)
    click.echo(f"  {'TOTAL':<36}  {total:>6}")
    click.echo()


# ──────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────

if __name__ == "__main__":
    cli()
