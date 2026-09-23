"""Empirical scalability benchmark: BM25 + FAISS (exact vs approximate)
across corpus sizes well beyond the 500-doc production corpus.

Separate from the main pipeline on purpose -- writes to data/scale_bench/,
never touches data/corpus.jsonl, data/faiss.index, etc. (the artifacts the
report's evaluation numbers are based on). Pure search-infra timing, no
LLM calls, no API cost/quota involved.

Usage: uv run python -m rag.scalability_bench
"""

import json
import time

import faiss
import numpy as np
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from rag import config
from rag.data_prep import clean_text

BENCH_DIR = config.DATA_DIR / "scale_bench"
MAX_DOCS = 50_000
TIERS = [500, 2_000, 10_000, 50_000]
SAMPLE_QUERIES = [
    "government announces new policy on climate change",
    "police investigate murder case in court trial",
    "celebrity dies after long battle with cancer",
    "company reports quarterly earnings and stock price",
    "president meets foreign leaders to discuss trade",
    "scientists discover new evidence about ancient history",
    "sports team wins championship after dramatic final",
    "flooding and storm damage cause evacuations",
    "technology company launches new product",
    "study finds link between diet and health outcomes",
]


def _tokenize(text: str) -> list:
    return text.lower().split()


def load_bench_corpus() -> list:
    """Load MAX_DOCS cleaned article texts from the full train split."""
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = BENCH_DIR / f"texts_{MAX_DOCS}.jsonl"
    if cache_path.exists():
        print(f"Using cached bench corpus: {cache_path}")
        texts = []
        with open(cache_path) as f:
            for line in f:
                texts.append(json.loads(line)["text"])
        return texts

    print(f"Loading up to {MAX_DOCS} articles from cnn_dailymail train split ...")
    dataset = load_dataset(config.HF_DATASET_NAME, config.HF_DATASET_VERSION, split="train")
    dataset = dataset.select(range(min(MAX_DOCS * 2, len(dataset))))

    texts = []
    with open(cache_path, "w") as f:
        for row in dataset:
            text = clean_text(row["article"])
            if len(text.split()) < 40:
                continue
            texts.append(text)
            f.write(json.dumps({"text": text}) + "\n")
            if len(texts) >= MAX_DOCS:
                break
    print(f"Loaded {len(texts)} articles -> {cache_path}")
    return texts


def build_embeddings(texts: list) -> np.ndarray:
    cache_path = BENCH_DIR / f"embeddings_{len(texts)}.npy"
    if cache_path.exists():
        print(f"Using cached embeddings: {cache_path}")
        return np.load(cache_path)

    print(f"Encoding {len(texts)} documents (one-time cost, reused for all tiers) ...")
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    t0 = time.time()
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    dt = time.time() - t0
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)
    print(f"Encoded {len(texts)} docs in {dt:.1f}s ({len(texts)/dt:.1f} docs/sec)")
    np.save(cache_path, embeddings)
    return embeddings


def bench_bm25(texts: list) -> dict:
    t0 = time.time()
    tokenized = [_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    build_time = time.time() - t0

    latencies = []
    for q in SAMPLE_QUERIES:
        t0 = time.time()
        bm25.get_scores(_tokenize(q))
        latencies.append(time.time() - t0)

    return {
        "build_seconds": round(build_time, 4),
        "avg_query_ms": round(sum(latencies) / len(latencies) * 1000, 3),
        "max_query_ms": round(max(latencies) * 1000, 3),
    }


def bench_faiss_exact(embeddings: np.ndarray, model, query_vecs: np.ndarray) -> dict:
    dim = embeddings.shape[1]
    t0 = time.time()
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    build_time = time.time() - t0

    latencies = []
    for qv in query_vecs:
        t0 = time.time()
        index.search(qv.reshape(1, -1), 10)
        latencies.append(time.time() - t0)

    return {
        "index_type": "IndexFlatIP (exact/brute-force)",
        "build_seconds": round(build_time, 4),
        "avg_query_ms": round(sum(latencies) / len(latencies) * 1000, 3),
        "max_query_ms": round(max(latencies) * 1000, 3),
    }, index


def bench_faiss_approx(embeddings: np.ndarray, query_vecs: np.ndarray, exact_index) -> dict:
    n = embeddings.shape[0]
    dim = embeddings.shape[1]
    # nlist ~ sqrt(n) is the standard IVF rule of thumb; needs enough training
    # vectors per cluster, so skip on tiers too small for it to make sense.
    nlist = max(4, int(n ** 0.5))
    if n < nlist * 40:
        return None

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

    t0 = time.time()
    index.train(embeddings)
    index.add(embeddings)
    build_time = time.time() - t0
    index.nprobe = max(1, nlist // 10)

    latencies = []
    recall_hits = 0
    for qv in query_vecs:
        qv2d = qv.reshape(1, -1)
        t0 = time.time()
        _, approx_ids = index.search(qv2d, 10)
        latencies.append(time.time() - t0)
        _, exact_ids = exact_index.search(qv2d, 10)
        recall_hits += len(set(approx_ids[0]) & set(exact_ids[0])) / 10

    return {
        "index_type": f"IndexIVFFlat (approximate, nlist={nlist}, nprobe={index.nprobe})",
        "build_seconds": round(build_time, 4),
        "avg_query_ms": round(sum(latencies) / len(latencies) * 1000, 3),
        "max_query_ms": round(max(latencies) * 1000, 3),
        "recall@10_vs_exact": round(recall_hits / len(query_vecs), 3),
    }


def run_benchmark() -> dict:
    texts = load_bench_corpus()
    embeddings = build_embeddings(texts)
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    query_vecs = model.encode(SAMPLE_QUERIES, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vecs)

    results = {}
    for tier in TIERS:
        if tier > len(texts):
            continue
        print(f"\n=== Tier: {tier} documents ===")
        tier_texts = texts[:tier]
        tier_embeddings = embeddings[:tier]

        bm25_result = bench_bm25(tier_texts)
        print(f"BM25: build={bm25_result['build_seconds']}s, avg_query={bm25_result['avg_query_ms']}ms")

        exact_result, exact_index = bench_faiss_exact(tier_embeddings, model, query_vecs)
        print(f"FAISS exact: build={exact_result['build_seconds']}s, avg_query={exact_result['avg_query_ms']}ms")

        approx_result = bench_faiss_approx(tier_embeddings, query_vecs, exact_index)
        if approx_result:
            print(
                f"FAISS approx: build={approx_result['build_seconds']}s, "
                f"avg_query={approx_result['avg_query_ms']}ms, "
                f"recall@10 vs exact={approx_result['recall@10_vs_exact']}"
            )
        else:
            print("FAISS approx: skipped (tier too small for meaningful IVF training)")

        results[tier] = {
            "bm25": bm25_result,
            "faiss_exact": exact_result,
            "faiss_approx": approx_result,
        }

    results_path = BENCH_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults -> {results_path}")
    return results


if __name__ == "__main__":
    run_benchmark()
