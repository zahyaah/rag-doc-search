"""Central configuration for the RAG document search system."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CORPUS_PATH = DATA_DIR / "corpus.jsonl"
TEST_SET_PATH = DATA_DIR / "test_set.jsonl"
FAISS_INDEX_PATH = DATA_DIR / "faiss.index"
BM25_INDEX_PATH = DATA_DIR / "bm25.pkl"
DOC_STORE_PATH = DATA_DIR / "doc_store.pkl"
EVAL_RESULTS_PATH = DATA_DIR / "eval_results.json"

# --- Corpus ---
CORPUS_SIZE = 500          # number of articles pulled from CNN/DailyMail
# NOTE: OpenRouter free-tier (":free" models) caps requests at 50/day for
# accounts with <$10 purchased credit (20 RPM ceiling too). Keep TEST_SET_SIZE
# small so retrieval + summary eval leaves headroom for live demo/debug calls
# in the same day. Source: https://openrouter.ai/docs/api-reference/limits
TEST_SET_SIZE = 15         # subset of corpus used for retrieval/summary eval
HF_DATASET_NAME = "cnn_dailymail"
HF_DATASET_VERSION = "3.0.0"

# --- Embeddings ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# --- Search ---
BM25_WEIGHT = 0.5
EMBEDDING_WEIGHT = 0.5
DEFAULT_TOP_N = 5

# --- LLM (via OpenRouter, OpenAI-compatible API) ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "http://localhost:8501")
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "rag-doc-search")

# --- Summarization ---
SUMMARY_LENGTHS = {
    "short": 50,
    "medium": 120,
    "long": 250,
}
