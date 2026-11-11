# Simple RAG

A small, readable Retrieval-Augmented Generation (RAG) pipeline in a single
Python file. Point it at a folder of `.txt`, `.md` and `.pdf` files, then ask
questions and get answers grounded in your documents, with citations.

```
documents -> chunks -> index -> retrieve top passages -> LLM answers from them
```

## Features

- **Ingest** `.txt`, `.md`, `.pdf` (heading/paragraph-aware chunking with overlap)
- **Retrieval**: TF-IDF keyword search, plus optional semantic search
  (sentence-transformers); results are merged with Reciprocal Rank Fusion
- **Three answer backends**, picked automatically:
  1. **Claude** (Anthropic API), if `ANTHROPIC_API_KEY` is set
  2. **Ollama** (free, local), if an Ollama server is running
  3. **Extractive**: just prints the matching passages (no key, no cost)
- Cited answers (`[1]`, `[2]`) and a source list; says "I don't know" when the
  documents don't cover the question
- No vector database needed; the index is one local file

## Quick start

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

python rag.py ingest docs/        # build the index (sample docs included)
python rag.py ask "What are the phases of incident response?"
python rag.py chat                # interactive mode
```

Put your own files in `docs/` (or any folder) and re-run `ingest` whenever they
change.

## Choose how answers are generated

### Option A: Claude (Anthropic API)

1. Create an account at <https://console.anthropic.com> and add a few dollars
   of credit under **Settings -> Billing**. A Claude.ai Pro/Max chat
   subscription does **not** include API access; they are billed separately.
2. **Settings -> API Keys -> Create Key**, then copy it (shown once).
3. Provide it to the script, either way:
   ```bash
   export ANTHROPIC_API_KEY="sk-ant-..."      # this terminal only
   ```
   or copy `.env.example` to `.env` and paste the key there (`.env` is
   git-ignored).
4. Optional, cheaper model: `export RAG_MODEL="claude-haiku-5-5"`

### Option B: Ollama (free, runs locally)

1. Install from <https://ollama.com/download> (or `brew install ollama`) and
   start it (open the app, or run `ollama serve`).
2. Pull a model: `ollama pull llama3.2` (about 2 GB). For better answers:
   `ollama pull llama3.1:8b` or `qwen2.5:7b` (8 GB+ RAM recommended).
3. Run `python rag.py ask "..."`. It is detected automatically when no
   Anthropic key is set.

### Option C: No LLM at all

Do nothing. The script prints the most relevant passages ("extractive mode").

### Force a backend

```bash
export RAG_BACKEND=ollama      # auto | anthropic | ollama | extractive
```

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | none | Enables Claude |
| `RAG_BACKEND` | `auto` | `auto`, `anthropic`, `ollama`, `extractive` |
| `RAG_MODEL` | `claude-sonnet-5-5` | Claude model name |
| `OLLAMA_MODEL` | `llama3.2` | Must be pulled first |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |

Chunk size, overlap and `TOP_K` are constants at the top of `rag.py`.

## Better retrieval (optional)

Keyword search only matches words that appear in the documents. For
meaning-based matching, install embeddings and re-ingest:

```bash
pip install sentence-transformers
python rag.py ingest docs/
```

The first run downloads a small embedding model (`all-MiniLM-L6-v2`).

## Tests

```bash
pip install pytest
pytest -q
```

## How it works

1. **Chunking**: documents are split on headings and blank lines, then into
   ~180-word windows with 40 words of overlap.
2. **Indexing**: TF-IDF (1-2 grams) and, if available, dense embeddings.
3. **Retrieval**: both rankings are fused (Reciprocal Rank Fusion) and the top
   `k` passages are returned; lexical-only mode drops passages with no
   term overlap.
4. **Generation**: the passages are sent to the LLM with an instruction to
   answer only from them and cite `[n]`.

## Security notes

- Never commit your API key. `.env` and `index.pkl` are in `.gitignore`.
- `index.pkl` is a Python pickle. Only load index files you built yourself;
  loading an untrusted pickle can execute arbitrary code.
- With the Claude backend, retrieved passages are sent to Anthropic's API. Use
  Ollama or extractive mode for documents that must stay on your machine.
- Documents are treated as untrusted text: retrieved content could contain
  instructions aimed at the model, so treat answers as aids, not authority.

## Project layout

```
rag.py                 the whole pipeline (ingest / ask / chat)
docs/                  sample documents
tests/test_rag.py      unit tests
requirements.txt
.env.example           config template
.github/workflows/     CI (runs pytest)
```

## Ideas for next steps

- Swap the pickle index for Chroma, FAISS or pgvector
- Add a reranker, or a simple web UI (Streamlit / Gradio)
- Metadata filters and incremental re-indexing

## License

MIT, see [LICENSE](LICENSE).
