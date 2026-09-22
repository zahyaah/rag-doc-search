"""Evaluation: retrieval accuracy (Recall@K, MRR) and summary quality (ROUGE).

Retrieval eval:
  For each document in the held-out test subset, an LLM generates a natural
  -language query that the document should ideally answer. We then run that
  query through the hybrid searcher and check whether the source document
  comes back in the top-K results.

Summary eval:
  For the same test documents, we summarize each document with our
  summarizer and compare the output to CNN/DailyMail's human-written
  reference summary ("highlights") using ROUGE-1/2/L.
"""

import json
import time

from rouge_score import rouge_scorer
from tqdm import tqdm

from rag import config
from rag.data_prep import load_jsonl
from rag.llm_client import chat_completion, LLMError
from rag.search import HybridSearcher
from rag.summarize import summarize_documents

QUERY_GEN_PATH = config.DATA_DIR / "eval_queries.jsonl"


def generate_query_for_doc(doc: dict) -> str:
    prompt = (
        "Read this news article and write ONE short, natural search query "
        "(5-12 words) that a person would type if they were looking for "
        "exactly this article. Return only the query text, nothing else.\n\n"
        f"Article:\n{doc['text'][:2000]}"
    )
    return chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=40,
        temperature=0.5,
    )


def generate_eval_queries(force: bool = False) -> list:
    """Generate (or load cached) queries for every doc in the test set."""
    if QUERY_GEN_PATH.exists() and not force:
        print(f"Using cached eval queries: {QUERY_GEN_PATH}")
        return load_jsonl(QUERY_GEN_PATH)

    test_docs = load_jsonl(config.TEST_SET_PATH)
    results = []
    for doc in tqdm(test_docs, desc="Generating eval queries"):
        try:
            query = generate_query_for_doc(doc)
        except LLMError as e:
            print(f"  [warn] query gen failed for {doc['doc_id']}: {e}")
            continue
        results.append({"doc_id": doc["doc_id"], "query": query, "text": doc["text"], "reference_summary": doc["reference_summary"]})
        time.sleep(1.5)  # be polite to the free-tier rate limit

    with open(QUERY_GEN_PATH, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    return results


def evaluate_retrieval(eval_queries: list, k_values=(1, 3, 5, 10)) -> dict:
    searcher = HybridSearcher()
    max_k = max(k_values)

    hits_at_k = {k: 0 for k in k_values}
    reciprocal_ranks = []
    per_query_results = []

    for item in tqdm(eval_queries, desc="Evaluating retrieval"):
        results = searcher.search(item["query"], top_n=max_k)
        retrieved_ids = [r["doc_id"] for r in results]

        if item["doc_id"] in retrieved_ids:
            rank = retrieved_ids.index(item["doc_id"]) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            rank = None
            reciprocal_ranks.append(0.0)

        for k in k_values:
            if rank is not None and rank <= k:
                hits_at_k[k] += 1

        per_query_results.append(
            {"doc_id": item["doc_id"], "query": item["query"], "found_at_rank": rank}
        )

    n = len(eval_queries)
    metrics = {
        f"recall@{k}": hits_at_k[k] / n for k in k_values
    }
    metrics["mrr"] = sum(reciprocal_ranks) / n
    metrics["n_queries"] = n
    return {"metrics": metrics, "per_query": per_query_results}


def evaluate_summaries(eval_queries: list, length: str = "medium", max_docs: int = None) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    subset = eval_queries[:max_docs] if max_docs else eval_queries
    rouge1, rouge2, rougeL = [], [], []
    per_doc_results = []

    for item in tqdm(subset, desc="Evaluating summaries"):
        try:
            summary = summarize_documents(
                [{"title": item["doc_id"], "text": item["text"]}], length=length
            )
        except LLMError as e:
            print(f"  [warn] summarize failed for {item['doc_id']}: {e}")
            continue

        scores = scorer.score(item["reference_summary"], summary)
        rouge1.append(scores["rouge1"].fmeasure)
        rouge2.append(scores["rouge2"].fmeasure)
        rougeL.append(scores["rougeL"].fmeasure)

        per_doc_results.append(
            {
                "doc_id": item["doc_id"],
                "generated_summary": summary,
                "reference_summary": item["reference_summary"],
                "rouge1_f": scores["rouge1"].fmeasure,
                "rouge2_f": scores["rouge2"].fmeasure,
                "rougeL_f": scores["rougeL"].fmeasure,
            }
        )
        time.sleep(1.5)

    n = len(per_doc_results) or 1
    metrics = {
        "rouge1_f_avg": sum(rouge1) / n,
        "rouge2_f_avg": sum(rouge2) / n,
        "rougeL_f_avg": sum(rougeL) / n,
        "n_docs": len(per_doc_results),
    }
    return {"metrics": metrics, "per_doc": per_doc_results}


def run_full_evaluation(summary_sample_size: int = 8) -> dict:
    eval_queries = generate_eval_queries()
    retrieval_results = evaluate_retrieval(eval_queries)
    summary_results = evaluate_summaries(eval_queries, max_docs=summary_sample_size)

    report = {
        "retrieval": retrieval_results["metrics"],
        "summarization": summary_results["metrics"],
        "retrieval_per_query": retrieval_results["per_query"],
        "summarization_per_doc": summary_results["per_doc"],
    }

    with open(config.EVAL_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== Retrieval metrics ===")
    for k, v in retrieval_results["metrics"].items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n=== Summarization metrics (ROUGE F1) ===")
    for k, v in summary_results["metrics"].items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    print(f"\nFull results -> {config.EVAL_RESULTS_PATH}")
    return report


if __name__ == "__main__":
    run_full_evaluation()
