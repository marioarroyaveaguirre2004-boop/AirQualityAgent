"""Recuperación top-k sobre el corpus con umbral de relevancia.

Backend denso (embeddings multilingües, similitud coseno) si los vectores y el modelo están
disponibles; si no (sin internet para bajar el modelo, p. ej.), BM25 sobre los mismos chunks.
La búsqueda densa es un producto interno exacto en NumPy sobre `embeddings.npy` (los mismos
vectores del IndexFlatIP de FAISS, con resultados idénticos). FAISS solo se usa al construir el
índice: cargar FAISS y PyTorch en el mismo proceso tumba la app en macOS (segmentation fault,
dos runtimes de OpenMP). `RAG_BACKEND=bm25` en el `.env` fuerza BM25.
Si ningún fragmento supera el umbral se devuelve una lista vacía y `encontrado=False`, para que
el agente diga explícitamente que no hay documento que respalde la respuesta.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from functools import lru_cache

import numpy as np

from src.config import EMBEDDING_MODEL, INDEX_DIR, RAG_MIN_SCORE

# Deben fijarse antes de importar torch.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

BM25_MIN_SCORE = 3.0
_STOP = set("""a al algo como con de del el en es esta este la las lo los más mas o para por que qué
se si sin sobre su sus un una uno y ya cual cuál cuales cuando cuándo donde dónde hay me mi""".split())


def _tokens(text: str) -> list[str]:
    t = unicodedata.normalize("NFKD", text.lower())
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    return [w for w in re.findall(r"[a-z0-9.,]+", t) if w not in _STOP and len(w) > 1]


class Retriever:
    def __init__(self) -> None:
        path = INDEX_DIR / "chunks.jsonl"
        if not path.exists():
            raise FileNotFoundError("No existe data/index/chunks.jsonl; corre `python -m src.rag.ingest`.")
        self.chunks = [json.loads(line) for line in path.open(encoding="utf-8")]
        self.backend = "bm25"
        self._model = self._emb = None
        emb_path = INDEX_DIR / "embeddings.npy"
        if emb_path.exists() and os.getenv("RAG_BACKEND", "").lower() != "bm25":
            try:
                from sentence_transformers import SentenceTransformer

                self._emb = np.load(emb_path).astype("float32")
                if self._emb.shape[0] != len(self.chunks):
                    raise ValueError("embeddings.npy no coincide con chunks.jsonl; corre el ingest")
                self._model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
                self.backend = "denso"
            except Exception as e:  # noqa: BLE001
                print(f"[aviso] embeddings no disponibles ({e}); uso BM25.")
        from rank_bm25 import BM25Okapi

        self._bm25 = BM25Okapi([_tokens(c["texto"]) for c in self.chunks])

    def search(self, query: str, k: int = 4) -> dict:
        """Devuelve {'encontrado', 'backend', 'resultados': [{fuente, pagina, texto, score}]}."""
        k = max(1, min(int(k), 8))
        if self.backend == "denso":
            q = self._model.encode([query], normalize_embeddings=True).astype("float32")[0]
            sims = self._emb @ q  # coseno: vectores normalizados
            top = np.argsort(-sims)[:k]
            pairs = [(int(i), float(sims[i])) for i in top]
            thr = RAG_MIN_SCORE
        else:
            sc = self._bm25.get_scores(_tokens(query))
            top = np.argsort(sc)[::-1][:k]
            pairs = [(int(i), float(sc[i])) for i in top]
            thr = BM25_MIN_SCORE
        results = [{"fuente": self.chunks[i]["fuente"], "titulo": self.chunks[i]["titulo"],
                    "pagina": self.chunks[i]["pagina"], "texto": self.chunks[i]["texto"],
                    "score": round(s, 3)} for i, s in pairs if s >= thr]
        return {"encontrado": bool(results), "backend": self.backend, "umbral": thr,
                "resultados": results}


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever()


if __name__ == "__main__":
    import sys

    r = get_retriever()
    for q in sys.argv[1:] or ["límite máximo permisible de PM2.5 en 24 horas"]:
        out = r.search(q)
        print(f"\n### {q}  [{out['backend']}]")
        for x in out["resultados"]:
            print(f"- {x['fuente']}, p. {x['pagina']} ({x['score']}): {x['texto'][:160]}...")
