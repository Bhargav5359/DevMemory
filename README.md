# DevMemory (devmem) 🧠⚡

> **Zero-cost, 100% offline local RAG CLI tool that indexes and retrieves your developer context** — shell history, VS Code workspaces, and online platform problem solutions (LeetCode, HackerRank) using ChromaDB and Ollama.

---

## ✨ Features

- **🔒 100% Local & Private:** No API keys, zero cloud latency, zero cost.
- **⚡ Fast Vector Retrieval:** Powered by persistent [ChromaDB](https://www.trychroma.com/) vector storage (`~/.devmem_db`) with cosine similarity.
- **🧠 Semantic Understanding:** Uses `nomic-embed-text` for embeddings and `llama3.2` for concise reasoning and answers.
- **🛠️ Shell History Sync (`devmem sync`):** Automatically parses `~/.zsh_history` or `~/.bash_history`, sanitizes timestamps, strips trivial commands, and indexes unique invocations.
- **💻 Workspace Code Sync (`devmem sync-code`):** Recursively chunks source files (`.py`, `.ts`, `.js`, `.go`, `.rs`, `.cpp`, `.java`, `.md`) in 50-line blocks, automatically ignoring `node_modules`, `.git`, `venv`, and build caches.
- **🧩 Platform Solutions Sync (`devmem sync-leetcode`):** Indexes your local repositories of problem solutions (LeetCode, HackerRank, etc.) with problem name metadata.
- **💬 Context-Aware Q&A (`devmem ask`):** Formats retrieved snippets into labelled context blocks (`[Terminal Command]`, `[VS Code | file (Line X)]`, `[LeetCode/Platform | problem]`) and provides exact code/command solutions with 1-sentence explanations.
- **📊 Storage Statistics (`devmem stats`):** Quick breakdown of indexed memories across all sources.
- **🛡️ Deduplication:** Content-hashed unique IDs (`doc_<sha256[:16]>`) prevent redundant embeddings and vector bloat.

---

## 🚀 Quick Start

### 1. Prerequisites

- **Python 3.10+**
- **[Ollama](https://ollama.com/)** installed and running

Pull the required models:
```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

Ensure the Ollama daemon is active:
```bash
ollama serve
```

### 2. Installation

Clone the repository and install dependencies:
```bash
git clone https://github.com/Noone5359/DevMemory.git
cd DevMemory
pip install -r requirements.txt
```

---

## 📖 Usage & CLI Commands

### 1. Sync Shell History
Indexes recent commands from `~/.zsh_history` or `~/.bash_history`:
```bash
python devmem.py sync --limit 500
```

### 2. Sync Code Workspaces
Indexes source code from your projects in 50-line blocks:
```bash
python devmem.py sync-code /path/to/your/project
```

### 3. Sync LeetCode / Coding Platform Solutions
Indexes solved algorithmic challenges:
```bash
python devmem.py sync-leetcode /path/to/leetcode-repo
```

### 4. Query Developer Memory
Ask questions about commands, code snippets, or solutions:
```bash
# Query terminal commands
python devmem.py ask "how do I run docker compose in background"

# Query code or algorithms
python devmem.py ask "how did I implement binary search or two sum" --top-k 3
```

### 5. View Memory Stats
Display total count of indexed items grouped by source type:
```bash
python devmem.py stats
```

---

## 🏗️ Architecture

```
Developer Contexts
 ├── ~/.zsh_history / ~/.bash_history
 ├── VS Code Workspace Files (py, ts, js, go, rs, cpp, java, md)
 └── LeetCode / Platform Repo
       │
       ▼
  Sanitization & Chunking (50 lines / doc_hash)
       │
       ▼
  Ollama: nomic-embed-text (Local Embeddings)
       │
       ▼
  ChromaDB (~/.devmem_db) [dev_memory collection]
       │
       ▲
  Query Embedding + Top-K Cosine Search
       │
       ▼
  Labelled Context Assembly
  ([Terminal Command] / [VS Code] / [LeetCode])
       │
       ▼
  Ollama: llama3.2 (Concise Answer + 1-Sentence Source Attribution)
```

---

## 📄 License

MIT License. Free and open source for developers everywhere.
