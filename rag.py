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

HOW TO READ THIS FILE (the order matches the lecture)
  1. Settings         the knobs: chunk size, top-k, which models to use
  2. Loading/chunking turn files into small passages ("chunks")         [indexing]
  3. Index            TF-IDF + embeddings, then search + fusion         [retrieval]
  4. Generation       build a grounded prompt, call a language model    [RAG]
  5. main()           the command-line: ingest / ask / chat

THE WHOLE PIPELINE IN ONE PICTURE
  ingest:  files -> read_file -> chunk_text -> Chunk objects -> Index (saved to index.pkl)
  ask:     question -> Index.search (top-k chunks) -> answer() builds prompt
                    -> ask_claude / ask_ollama / extractive -> show() prints answer + sources
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

# python-dotenv is optional. If it is installed, this copies the lines of a local
# ".env" file (for example ANTHROPIC_API_KEY=...) into the environment, so you do
# not have to type `export ...` in every terminal. If it is missing we just move on.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from sklearn.feature_extraction.text import TfidfVectorizer

# ----------------------------------------------------------------------------
# 1. SETTINGS: change these to run experiments (chunk size sweep, top-k sweep...)
# ----------------------------------------------------------------------------
INDEX_PATH = Path("index.pkl")   # where the built index is saved between runs
CHUNK_WORDS = 180      # target chunk size (words). ~235 tokens, near the embedder's limit
CHUNK_OVERLAP = 40     # overlap between consecutive chunks (words); protects sentence edges
TOP_K = 4              # how many chunks go to the language model as evidence
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # 384-number vectors, runs on CPU
CLAUDE_MODEL = os.environ.get("RAG_MODEL", "claude-haiku-5-5")   # override via .env
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")         # small local model (~3B)


@dataclass
class Chunk:
    """One searchable passage plus where it came from (needed for citations)."""
    text: str      # the passage itself
    source: str    # file it came from, relative to the docs folder
    idx: int       # chunk number inside that file (shown as "file#idx")


# ----------------------------------------------------------------------------
# 2. LOADING AND CHUNKING  (lecture: document indexing)
# ----------------------------------------------------------------------------
def read_file(path: Path) -> str:
    """Return the plain text of a .txt, .md or .pdf file."""
    if path.suffix.lower() == ".pdf":
        # imported here so people who never use PDFs do not need pypdf installed
        from pypdf import PdfReader
        # extract_text() can return None for scanned pages (no text layer), hence `or ""`
        return "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    # errors="ignore" skips undecodable bytes instead of crashing on odd files
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, size=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    """Paragraph-aware sliding window over words.

    Why chunk? An embedding model reads a limited amount of text, one vector for a
    whole book would blur every topic together, and the prompt has limited room.
    Why overlap? So an idea cut at a chunk boundary still appears whole in one chunk.
    """
    out = []
    # Step 1: split on markdown headings and blank lines, so a window never mixes
    # two sections. (An early version skipped this and made one chunk per file.)
    #   \n(?=#{1,6} )  = a newline that is followed by "# ", "## " ... "###### "
    #   \n\s*\n        = a blank line
    for section in re.split(r"\n(?=#{1,6} )|\n\s*\n", text):
        words = section.split()
        if not words:          # skip empty sections
            continue
        # Step 2: slide a window of `size` words, moving forward by size - overlap.
        # With size 180 and overlap 40 the step is 140: chunks start at word 0, 140, 280...
        step = max(1, size - overlap)   # max(1, ...) guards against an infinite loop
        out += [" ".join(words[i:i + size]) for i in range(0, len(words), step)]
    return out


def load_chunks(folder: Path):
    """Read every supported file under `folder` and cut it into Chunk objects."""
    chunks = []
    for p in sorted(folder.rglob("*")):    # rglob = look in sub-folders too; sorted = stable order
        if p.suffix.lower() in {".txt", ".md", ".pdf"} and p.is_file():
            # i restarts at 0 for each file, so the citation reads "cybersecurity.md#1"
            for i, t in enumerate(chunk_text(read_file(p))):
                chunks.append(Chunk(t, str(p.relative_to(folder)), i))
    return chunks


# ----------------------------------------------------------------------------
# 3. INDEX  (lecture: TF-IDF/BM25, embeddings, semantic search, hybrid + RRF)
# ----------------------------------------------------------------------------
class Index:
    """Holds the chunks and two ways to search them: by words (TF-IDF) and by meaning."""

    def __init__(self, chunks):
        self.chunks = chunks
        texts = [c.text for c in chunks]

        # LEXICAL side: TF-IDF turns each chunk into a vector of word weights.
        #   stop_words="english"  ignore filler words (the, and, of ...)
        #   ngram_range=(1, 2)    index single words AND word pairs ("zero trust")
        #   sublinear_tf=True     use 1 + ln(tf), so repeating a word 10x is not 10x better
        # Note: no stemming, so "email" will not match "emails".
        self.tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                     sublinear_tf=True)
        self.tfidf_mat = self.tfidf.fit_transform(texts)   # shape: (chunks, vocabulary)

        # SEMANTIC side: one 384-number embedding per chunk. Optional: if the library
        # is missing or the model cannot be downloaded, we keep going with TF-IDF only
        # (this is the "graceful degradation" idea: the program always runs).
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
        # imported lazily: sentence-transformers is large and optional.
        # (Known limitation: this builds the model again on every call, so queries are
        #  slow. A good exercise is to cache the model after the first load.)
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(EMBED_MODEL)

    def _embed(self, texts):
        # normalize_embeddings=True scales every vector to length 1, so the dot product
        # of two vectors equals their cosine similarity (which is why `@` is used below)
        return self._embedder().encode(texts, normalize_embeddings=True,
                                       show_progress_bar=False)

    def search(self, query, k=TOP_K):
        """Return the k best chunks as (chunk, fused_score, tfidf_similarity) tuples."""
        rankings = []   # one ranked list of chunk numbers per retriever

        # Ranking 1 (lexical): TF-IDF vector of the query times every chunk vector
        # = cosine similarity with every chunk. argsort(-sims) lists chunk numbers best first.
        sims = (self.tfidf.transform([query]) @ self.tfidf_mat.T).toarray()[0]
        rankings.append(np.argsort(-sims))

        # Ranking 2 (semantic): embed the query with the SAME model, then one matrix
        # product gives the similarity to all chunks at once.
        if self.dense is not None:
            q = self._embed([query])[0]
            rankings.append(np.argsort(-(self.dense @ q)))

        # Reciprocal Rank Fusion (RRF): merge the lists using ranks only, so the two
        # score scales never need to be compared. A chunk at position `rank` in a list
        # earns 1 / (60 + rank); it earns from each list it appears in, so a chunk that
        # is good in BOTH lists rises to the top. 60 is the usual constant; only the
        # best 50 of each list are used. (Here rank starts at 0; the paper starts at 1.)
        scores = {}
        for r in rankings:
            for rank, i in enumerate(r[:50]):
                scores[i] = scores.get(i, 0) + 1.0 / (60 + rank)
        top = sorted(scores, key=scores.get, reverse=True)[:k]

        # each hit: (the chunk, its fused RRF score, its TF-IDF similarity)
        hits = [(self.chunks[i], scores[i], sims[i]) for i in top]

        # Lexical-only safety net: a chunk sharing no words with the question is not a
        # real match, even if it ranks in the top k. Drop it, so off-topic questions
        # return nothing instead of a random chunk (the model is then never called).
        if self.dense is None:  # lexical only: drop chunks with no term overlap
            hits = [h for h in hits if h[2] > 0]
        return hits

    def save(self, path=INDEX_PATH):
        # Pickle stores the whole object (chunks, TF-IDF model, embeddings) in one file.
        # SECURITY: loading a pickle can run code, so only open files you built yourself.
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path=INDEX_PATH):
        with open(path, "rb") as f:
            return pickle.load(f)  # only load indexes you built yourself


# ----------------------------------------------------------------------------
# 4. GENERATION  (lecture: RAG, grounded prompting, RAG vs SLM)
# ----------------------------------------------------------------------------
# The "contract" given to the model: use only the supplied evidence, cite it by
# number, and admit when the evidence does not contain the answer. This one prompt
# is what turns a chatbot into a grounded, citing assistant.
SYSTEM = (
    "You answer questions using ONLY the provided context passages. "
    "Cite passages inline like [1], [2]. If the context does not contain the "
    "answer, say you don't know based on the documents. Be concise."
)


def ollama_up() -> bool:
    """True if a local Ollama server answers on its default port."""
    try:
        urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=2)   # lists installed models
        return True
    except Exception:
        return False


def ask_ollama(context, question):
    """Ask a local model through Ollama's HTTP API (free, private, no key)."""
    body = json.dumps({
        "model": OLLAMA_MODEL, "stream": False,    # stream False = wait for the full reply
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user",
                      "content": f"Context:\n{context}\n\nQuestion: {question}"}],
    }).encode()
    req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:   # local models can be slow
        return json.load(r)["message"]["content"]


def ask_claude(context, question):
    """Ask Claude through the Anthropic API (needs ANTHROPIC_API_KEY and API credit)."""
    import anthropic   # imported here so Ollama-only users do not need the package
    # Anthropic() reads the key from the ANTHROPIC_API_KEY environment variable
    msg = anthropic.Anthropic().messages.create(
        model=CLAUDE_MODEL, max_tokens=800, system=SYSTEM,
        messages=[{"role": "user",
                   "content": f"Context:\n{context}\n\nQuestion: {question}"}],
    )
    # the reply can start with a thinking block, so join only the text blocks
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def pick_backend():
    """Choose who writes the answer. Retrieval is identical for all three."""
    b = os.environ.get("RAG_BACKEND", "auto").lower()
    if b != "auto":                # the user forced a choice
        return b
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"         # best quality, data goes to the cloud
    return "ollama" if ollama_up() else "extractive"   # local, else just print passages


def answer(index: Index, question: str, k=TOP_K):
    """The RAG step: retrieve evidence, build the prompt, get an answer.

    Returns (answer_text, hits) so the caller can also print the sources.
    """
    hits = index.search(question, k)
    if not hits:
        # No evidence found: refuse without calling any model. Cheapest and most
        # reliable way to stop a confident answer made from nothing (hallucination).
        return "I couldn't find anything relevant in the documents.", hits

    # Number each chunk [1], [2]... and label its origin, so the model can cite
    # them and a reader can check the file and chunk.
    context = "\n\n".join(f"[{n}] ({c.source}#{c.idx})\n{c.text}"
                          for n, (c, _, _) in enumerate(hits, 1))
    backend = pick_backend()
    try:
        if backend == "anthropic":
            return ask_claude(context, question), hits
        if backend == "ollama":
            return ask_ollama(context, question), hits
    except Exception as e:
        # Any failure (bad key, no credit, server down) falls through to extractive
        # mode, so the user still sees the retrieved passages instead of a crash.
        print(f"[warn] {backend} failed ({type(e).__name__}: {e}); "
              "showing passages instead", file=sys.stderr)
    # extractive mode: no model, show the evidence itself (useful to debug retrieval)
    return ("(extractive mode: no generated answer. Set ANTHROPIC_API_KEY or start "
            "Ollama; if one is set, see the warning above)\n\n" + context), hits


def show(ans, hits):
    """Print the answer, then the list of sources it was built from."""
    print(ans)
    print("\nSources:")
    for n, (c, _, _) in enumerate(hits, 1):
        print(f"  [{n}] {c.source} (chunk {c.idx})")


# ----------------------------------------------------------------------------
# 5. COMMAND LINE
# ----------------------------------------------------------------------------
def main():
    # Three sub-commands:  ingest <folder> | ask "<question>" [-k N] | chat
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("ingest"); s.add_argument("folder")
    s = sub.add_parser("ask"); s.add_argument("question"); s.add_argument("-k", type=int, default=TOP_K)
    sub.add_parser("chat")
    a = ap.parse_args()

    if a.cmd == "ingest":
        # Offline half of RAG: read -> chunk -> index -> save. Run again after docs change.
        chunks = load_chunks(Path(a.folder))
        if not chunks:
            sys.exit("No .txt/.md/.pdf content found.")
        idx = Index(chunks); idx.save()
        # prints the mode so you can see whether embeddings were available
        print(f"Indexed {len(chunks)} chunks from {len({c.source for c in chunks})} files "
              f"({'hybrid' if idx.dense is not None else 'tf-idf'}).")
    elif a.cmd == "ask":
        # Online half, one question: load the saved index, retrieve, answer, print.
        show(*answer(Index.load(), a.question, a.k))
    else:
        # chat: load the index once, then answer questions until Enter on an empty
        # line is not needed; Ctrl-C or Ctrl-D ends the loop.
        # (Each question is searched on its own: follow-ups like "and the second one?"
        #  lose their topic. Query rewriting would fix that.)
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
