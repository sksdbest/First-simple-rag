#!/usr/bin/env python3
"""Minimal RAG: ingest -> chunk -> index -> retrieve -> generate.

Usage:
    python rag.py ingest docs/            # build index from .txt/.md/.pdf files
    python rag.py ask "your question"     # retrieve + answer
    python rag.py chat                    # interactive loop

Retrieval: hybrid of TF-IDF (lexical) and dense embeddings (semantic) fused with
Reciprocal Rank Fusion. Dense embeddings are used if `sentence-transformers` is
installed; otherwise it falls back to TF-IDF only (no downloads, no keys).
Generation backends (RAG_BACKEND = auto | anthropic | ollama | extractive):
  auto (default): Claude if ANTHROPIC_API_KEY is set, else a local Ollama server
  if one is running, else extractive mode (just prints the top passages).
"""
import argparse
import json
import os
import pickle
import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:  # optional: load ANTHROPIC_API_KEY etc. from a .env file
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from sklearn.feature_extraction.text import TfidfVectorizer

INDEX_PATH = Path("index.pkl")
CHUNK_WORDS = 180      # target chunk size (words)
CHUNK_OVERLAP = 40     # overlap between consecutive chunks (words)
TOP_K = 4
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CLAUDE_MODEL = os.environ.get("RAG_MODEL", "claude-sonnet-5-5")
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")


@dataclass
class Chunk:
    text: str
    source: str
    idx: int


# ---------- loading & chunking ----------
def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, size=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    """Paragraph-aware sliding window over words."""
    out = []
    # split on markdown headings / blank-line-separated sections first
    for section in re.split(r"\n(?=#{1,6} )|\n\s*\n", text):
        words = section.split()
        if not words:
            continue
        step = max(1, size - overlap)
        out += [" ".join(words[i:i + size]) for i in range(0, len(words), step)]
    return out


def load_chunks(folder: Path):
    chunks = []
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() in {".txt", ".md", ".pdf"} and p.is_file():
            for i, t in enumerate(chunk_text(read_file(p))):
                chunks.append(Chunk(t, str(p.relative_to(folder)), i))
    return chunks


# ---------- index ----------
class Index:
    def __init__(self, chunks):
        self.chunks = chunks
        texts = [c.text for c in chunks]
        self.tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                     sublinear_tf=True)
        self.tfidf_mat = self.tfidf.fit_transform(texts)
        self.dense = None
        self.embedder_name = None
        try:
            self.dense = self._embed(texts)
            self.embedder_name = EMBED_MODEL
        except Exception as e:  # not installed / offline -> lexical only
            print(f"[info] dense embeddings unavailable ({type(e).__name__}); "
                  "using TF-IDF only", file=sys.stderr)

    @staticmethod
    def _embedder():
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(EMBED_MODEL)

    def _embed(self, texts):
        return self._embedder().encode(texts, normalize_embeddings=True,
                                       show_progress_bar=False)

    def search(self, query, k=TOP_K):
        rankings = []
        sims = (self.tfidf.transform([query]) @ self.tfidf_mat.T).toarray()[0]
        rankings.append(np.argsort(-sims))
        if self.dense is not None:
            q = self._embed([query])[0]
            rankings.append(np.argsort(-(self.dense @ q)))
        # Reciprocal Rank Fusion
        scores = {}
        for r in rankings:
            for rank, i in enumerate(r[:50]):
                scores[i] = scores.get(i, 0) + 1.0 / (60 + rank)
        top = sorted(scores, key=scores.get, reverse=True)[:k]
        hits = [(self.chunks[i], scores[i], sims[i]) for i in top]
        if self.dense is None:  # lexical only: drop chunks with no term overlap
            hits = [h for h in hits if h[2] > 0]
        return hits

    def save(self, path=INDEX_PATH):
        # drop nothing heavy: pickle is fine for a local, trusted index
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path=INDEX_PATH):
        with open(path, "rb") as f:
            return pickle.load(f)  # only load indexes you built yourself


# ---------- generation ----------
SYSTEM = (
    "You answer questions using ONLY the provided context passages. "
    "Cite passages inline like [1], [2]. If the context does not contain the "
    "answer, say you don't know based on the documents. Be concise."
)


def ollama_up() -> bool:
    try:
        urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=2)
        return True
    except Exception:
        return False


def ask_ollama(context, question):
    body = json.dumps({
        "model": OLLAMA_MODEL, "stream": False,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user",
                      "content": f"Context:\n{context}\n\nQuestion: {question}"}],
    }).encode()
    req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["message"]["content"]


def ask_claude(context, question):
    import anthropic
    msg = anthropic.Anthropic().messages.create(
        model=CLAUDE_MODEL, max_tokens=800, system=SYSTEM,
        messages=[{"role": "user",
                   "content": f"Context:\n{context}\n\nQuestion: {question}"}],
    )
    # the reply can start with a thinking block, so join only the text blocks
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def pick_backend():
    b = os.environ.get("RAG_BACKEND", "auto").lower()
    if b != "auto":
        return b
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "ollama" if ollama_up() else "extractive"


def answer(index: Index, question: str, k=TOP_K):
    hits = index.search(question, k)
    if not hits:
        return "I couldn't find anything relevant in the documents.", hits
    context = "\n\n".join(f"[{n}] ({c.source}#{c.idx})\n{c.text}"
                          for n, (c, _, _) in enumerate(hits, 1))
    backend = pick_backend()
    try:
        if backend == "anthropic":
            return ask_claude(context, question), hits
        if backend == "ollama":
            return ask_ollama(context, question), hits
    except Exception as e:
        print(f"[warn] {backend} failed ({type(e).__name__}: {e}); "
              "showing passages instead", file=sys.stderr)
    return ("(extractive mode: no generated answer. Set ANTHROPIC_API_KEY or start "
            "Ollama; if one is set, see the warning above)\n\n" + context), hits


def show(ans, hits):
    print(ans)
    print("\nSources:")
    for n, (c, _, _) in enumerate(hits, 1):
        print(f"  [{n}] {c.source} (chunk {c.idx})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("ingest"); s.add_argument("folder")
    s = sub.add_parser("ask"); s.add_argument("question"); s.add_argument("-k", type=int, default=TOP_K)
    sub.add_parser("chat")
    a = ap.parse_args()

    if a.cmd == "ingest":
        chunks = load_chunks(Path(a.folder))
        if not chunks:
            sys.exit("No .txt/.md/.pdf content found.")
        idx = Index(chunks); idx.save()
        print(f"Indexed {len(chunks)} chunks from {len({c.source for c in chunks})} files "
              f"({'hybrid' if idx.dense is not None else 'tf-idf'}).")
    elif a.cmd == "ask":
        show(*answer(Index.load(), a.question, a.k))
    else:
        idx = Index.load()
        while True:
            try:
                q = input("\nask> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q:
                show(*answer(idx, q))


if __name__ == "__main__":
    main()
