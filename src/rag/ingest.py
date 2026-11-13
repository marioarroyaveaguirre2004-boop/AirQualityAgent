"""Ingesta del corpus normativo: PDF -> texto por página -> chunks con solapamiento -> índice.

- Cada chunk queda dentro de UNA página, así la cita [fuente, p. N] es exacta.
- Chunks de ~350 palabras (≈ 500 tokens) con 60 palabras de solapamiento.
- Índice denso: sentence-transformers multilingüe + FAISS (producto interno sobre vectores
  normalizados = similitud coseno). Si el modelo no se puede cargar, solo se guardan los chunks
  y el retriever usa BM25.
Metadatos de cada documento en data/corpus/sources.json (archivo, fuente, título, URL, fecha).

Uso: python -m src.rag.ingest
"""
from __future__ import annotations

import json
import re

import numpy as np
from pypdf import PdfReader

from src.config import CORPUS_DIR, EMBEDDING_MODEL, INDEX_DIR

CHUNK_WORDS = 350
OVERLAP_WORDS = 60
MIN_CHUNK_WORDS = 25


def clean_text(t: str) -> str:
    t = t.replace("­", "")                       # guiones suaves
    t = re.sub(r"-\n(\w)", r"\1", t)                  # palabras partidas por salto de línea
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()


def pdf_pages(path) -> list[tuple[int, str]]:
    reader = PdfReader(str(path))
    return [(i + 1, clean_text(p.extract_text() or "")) for i, p in enumerate(reader.pages)]


def chunk_page(text: str) -> list[str]:
    words = text.split()
    if len(words) < MIN_CHUNK_WORDS:
        return []
    step = CHUNK_WORDS - OVERLAP_WORDS
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start:start + CHUNK_WORDS]
        if len(piece) >= MIN_CHUNK_WORDS or not chunks:
            chunks.append(" ".join(piece))
        if start + CHUNK_WORDS >= len(words):
            break
    return chunks


def build_chunks() -> tuple[list[dict], list[dict]]:
    sources = json.loads((CORPUS_DIR / "sources.json").read_text(encoding="utf-8"))
    chunks, report = [], []
    for src in sources:
        path = CORPUS_DIR / src["archivo"]
        if not path.exists():
            report.append({"archivo": src["archivo"], "error": "no existe"})
            continue
        pages = pdf_pages(path)
        n_before = len(chunks)
        for page_no, text in pages:
            for ch in chunk_page(text):
                chunks.append({"id": len(chunks), "fuente": src["fuente"], "titulo": src["titulo"],
                               "pagina": page_no, "texto": ch})
        report.append({"archivo": src["archivo"], "paginas": len(pages),
                       "paginas_sin_texto": sum(1 for _, t in pages if len(t.split()) < MIN_CHUNK_WORDS),
                       "chunks": len(chunks) - n_before})
    return chunks, report


def build_dense_index(chunks: list[dict]) -> str:
    """Construye el índice denso (FAISS + embeddings.npy); devuelve 'faiss' o 'bm25'.

    La app busca sobre embeddings.npy con NumPy (mismo resultado que el IndexFlatIP) para no
    cargar FAISS junto a PyTorch en tiempo de consulta (segfault en macOS)."""
    try:
        import faiss
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        emb = model.encode([c["texto"] for c in chunks], batch_size=32, show_progress_bar=True,
                           normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        faiss.write_index(index, str(INDEX_DIR / "index.faiss"))
        np.save(INDEX_DIR / "embeddings.npy", emb)
        return "faiss"
    except Exception as e:  # noqa: BLE001 - sin modelo/red: queda BM25
        print(f"[aviso] índice denso no disponible ({e}); se usará BM25.")
        (INDEX_DIR / "index.faiss").unlink(missing_ok=True)
        (INDEX_DIR / "embeddings.npy").unlink(missing_ok=True)
        return "bm25"


def main() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    chunks, report = build_chunks()
    with open(INDEX_DIR / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    backend = build_dense_index(chunks)
    meta = {"backend": backend, "embedding_model": EMBEDDING_MODEL if backend == "faiss" else None,
            "n_chunks": len(chunks), "chunk_words": CHUNK_WORDS, "overlap_words": OVERLAP_WORDS,
            "documentos": report}
    (INDEX_DIR / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
