"""Data preparation: download, clean, and split the corpus.

Uses the CNN/DailyMail dataset (via HuggingFace `datasets`) because each
article ships with a human-written reference summary ("highlights"), which
we later use as ground truth for ROUGE evaluation.
"""

import json
import random
import re

from datasets import load_dataset
from tqdm import tqdm

from rag import config

random.seed(42)


def clean_text(text: str) -> str:
    """Normalize whitespace and strip CNN/DailyMail boilerplate artifacts."""
    text = re.sub(r"\(CNN\)\s*(?:--\s*)?", "", text)
    text = re.sub(r"By \. .*?\. PUBLISHED: .*?\. \|.*?UPDATED:.*?\.", "", text, flags=re.DOTALL)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_corpus() -> None:
    """Download the dataset, clean it, and write corpus.jsonl + test_set.jsonl."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {config.HF_DATASET_NAME} ({config.HF_DATASET_VERSION}) ...")
    dataset = load_dataset(
        config.HF_DATASET_NAME, config.HF_DATASET_VERSION, split="test"
    )
    dataset = dataset.select(range(min(config.CORPUS_SIZE * 2, len(dataset))))

    records = []
    for row in tqdm(dataset, desc="Cleaning articles"):
        article = clean_text(row["article"])
        highlights = clean_text(row["highlights"])
        if len(article.split()) < 40:
            continue  # skip near-empty / malformed rows
        records.append(
            {
                "doc_id": row["id"],
                "title": article.split(".")[0][:120],
                "text": article,
                "reference_summary": highlights,
            }
        )
        if len(records) >= config.CORPUS_SIZE:
            break

    corpus_records = records

    # Test set is a SUBSET of the corpus (per assignment spec), not held out:
    # each test doc must remain searchable so retrieval-accuracy eval is meaningful.
    test_records = random.sample(corpus_records, k=min(config.TEST_SET_SIZE, len(corpus_records)))

    with open(config.CORPUS_PATH, "w", encoding="utf-8") as f:
        for rec in corpus_records:
            f.write(json.dumps(rec) + "\n")

    with open(config.TEST_SET_PATH, "w", encoding="utf-8") as f:
        for rec in test_records:
            f.write(json.dumps(rec) + "\n")

    print(f"Corpus: {len(corpus_records)} docs -> {config.CORPUS_PATH}")
    print(f"Test set (subset of corpus, used for eval): {len(test_records)} docs -> {config.TEST_SET_PATH}")


def load_jsonl(path) -> list:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


if __name__ == "__main__":
    prepare_corpus()
