import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rag  # noqa: E402

DOCS = Path(__file__).resolve().parent.parent / "docs"


def test_chunking_splits_sections_and_overlaps():
    text = "# A\n" + "word " * 400 + "\n\n# B\nshort section"
    chunks = rag.chunk_text(text, size=100, overlap=20)
    assert len(chunks) > 3
    assert any(c.startswith("# B") for c in chunks)


def test_retrieval_finds_right_document(monkeypatch):
    monkeypatch.setattr(rag.Index, "_embedder", staticmethod(lambda: 1 / 0))
    idx = rag.Index(rag.load_chunks(DOCS))
    top = idx.search("when is wheat harvested", k=1)[0][0]
    assert top.source == "agro.txt"
    top = idx.search("incident response phases", k=1)[0][0]
    assert top.source == "cybersecurity.md"


def test_irrelevant_query_returns_nothing(monkeypatch):
    monkeypatch.setattr(rag.Index, "_embedder", staticmethod(lambda: 1 / 0))
    idx = rag.Index(rag.load_chunks(DOCS))
    assert idx.search("quantum chromodynamics") == []


def test_backend_selection(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("RAG_BACKEND", raising=False)
    monkeypatch.setattr(rag, "ollama_up", lambda: False)
    assert rag.pick_backend() == "extractive"
    monkeypatch.setattr(rag, "ollama_up", lambda: True)
    assert rag.pick_backend() == "ollama"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert rag.pick_backend() == "anthropic"
    monkeypatch.setenv("RAG_BACKEND", "ollama")
    assert rag.pick_backend() == "ollama"
