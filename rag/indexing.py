"""Build BM25 and FAISS embedding indexes over the corpus."""

import pickle

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from rag import config
from rag.data_prep import load_jsonl


def _tokenize(text: str) -> list:
    return text.lower().split()


def build_indexes() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    docs = load_jsonl(config.CORPUS_PATH)
    if not docs:
        raise RuntimeError(f"No documents found in {config.CORPUS_PATH}. Run data_prep first.")

    # --- BM25 (lexical / keyword search) ---
    print(f"Building BM25 index over {len(docs)} docs ...")
    tokenized_corpus = [_tokenize(d["text"]) for d in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(config.BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)

    # --- Dense embeddings (semantic search) ---
    print(f"Loading embedding model {config.EMBEDDING_MODEL_NAME} ...")
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    print("Encoding corpus ...")
    texts = [d["text"] for d in docs]
    embeddings = model.encode(
        texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True
    )
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)  # so inner product == cosine similarity

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(config.FAISS_INDEX_PATH))

    # --- Doc store: id -> metadata, needed to map index positions back to docs ---
    with open(config.DOC_STORE_PATH, "wb") as f:
        pickle.dump(docs, f)

    print(f"BM25 index -> {config.BM25_INDEX_PATH}")
    print(f"FAISS index -> {config.FAISS_INDEX_PATH} ({index.ntotal} vectors, dim={embeddings.shape[1]})")
    print(f"Doc store -> {config.DOC_STORE_PATH}")


if __name__ == "__main__":
    build_indexes()
