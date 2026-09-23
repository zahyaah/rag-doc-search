"""Build BM25 and FAISS embedding indexes over the corpus.

BM25 uses bm25s (numpy-vectorized), not rank_bm25 (pure Python loops) --
rank_bm25 was measured at 560ms/query at 50,000 docs vs bm25s's published
~400x throughput advantage at that scale (see REPORT.md §6 and
docs/decisions/0001-hybrid-retrieval.md). FAISS switches from exact
IndexFlatIP to approximate IndexIVFFlat once the corpus is large enough to
train it well (config.FAISS_IVF_MIN_DOCS) -- below that it stays exact,
so the default 500-doc corpus behaves identically to before.
"""

import pickle

import bm25s
import faiss
from sentence_transformers import SentenceTransformer

from rag import config
from rag.data_prep import load_jsonl


def build_indexes() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    docs = load_jsonl(config.CORPUS_PATH)
    if not docs:
        raise RuntimeError(f"No documents found in {config.CORPUS_PATH}. Run data_prep first.")
    n_docs = len(docs)

    # --- BM25 (lexical / keyword search) ---
    print(f"Building BM25 index (bm25s) over {n_docs} docs ...")
    texts = [d["text"] for d in docs]
    corpus_tokens = bm25s.tokenize(texts, stopwords="en", show_progress=False)
    bm25_retriever = bm25s.BM25()
    bm25_retriever.index(corpus_tokens, show_progress=False)
    # corpus=None: doc text/metadata already lives in doc_store.pkl below,
    # no need to duplicate it inside the BM25 index directory too.
    bm25_retriever.save(str(config.BM25_INDEX_PATH), corpus=None)

    # --- Dense embeddings (semantic search) ---
    print(f"Loading embedding model {config.EMBEDDING_MODEL_NAME} ...")
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    print("Encoding corpus ...")
    embeddings = model.encode(
        texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True
    )
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)  # so inner product == cosine similarity
    dim = embeddings.shape[1]

    if n_docs >= config.FAISS_IVF_MIN_DOCS:
        nlist = max(4, int(n_docs ** 0.5))
        print(f"Building approximate FAISS IndexIVFFlat (nlist={nlist}) -- corpus large enough to train it ...")
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = max(1, nlist // 10)
    else:
        print("Building exact FAISS IndexFlatIP (corpus too small to usefully train an ANN index) ...")
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

    faiss.write_index(index, str(config.FAISS_INDEX_PATH))

    # --- Doc store: id -> metadata, needed to map index positions back to docs ---
    with open(config.DOC_STORE_PATH, "wb") as f:
        pickle.dump(docs, f)

    print(f"BM25 index -> {config.BM25_INDEX_PATH}")
    print(f"FAISS index -> {config.FAISS_INDEX_PATH} ({index.ntotal} vectors, dim={dim})")
    print(f"Doc store -> {config.DOC_STORE_PATH}")


if __name__ == "__main__":
    build_indexes()
