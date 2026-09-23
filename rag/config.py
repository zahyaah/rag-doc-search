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
# bm25s saves to a directory (multiple files), not a single pickle.
# Source: https://github.com/xhluca/bm25s
BM25_INDEX_PATH = DATA_DIR / "bm25_index"
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

# How many top candidates to pull from EACH retriever (BM25, FAISS) before
# fusing scores. At corpus sizes where n <= CANDIDATE_K, this is equivalent
# to scoring the whole corpus (what the original small-corpus implementation
# did); at scale, this keeps per-query cost bounded by CANDIDATE_K rather
# than by corpus size -- both bm25s.retrieve() and faiss.search() are
# themselves sublinear/fast at top-k retrieval, so the O(n) cost of scoring
# the *entire* corpus per query (the original bottleneck measured in
# REPORT.md §6.1) is avoided entirely, not just made faster.
RETRIEVAL_CANDIDATE_K = 100

# Above this corpus size, build an approximate FAISS IndexIVFFlat instead of
# exact IndexFlatIP (see docs/decisions/0001-hybrid-retrieval.md). Matches
# the empirically-validated rule from rag/scalability_bench.py: IVF needs
# roughly n >= nlist*40 training vectors to be well-calibrated, and
# nlist ~= sqrt(n), so this is the smallest n where that holds.
FAISS_IVF_MIN_DOCS = 1600

# --- LLM provider ---
# Default: OpenRouter (documented decision, see docs/decisions/0003-llm-provider.md).
# Set LLM_PROVIDER=gemini in .env to switch to Google AI Studio's free tier
# instead (separate quota from OpenRouter's, useful when OpenRouter's 50/day
# cap is exhausted). Both speak the OpenAI-compatible chat/completions shape,
# so rag/llm_client.py needs no provider-specific branching beyond base URL,
# key, and a couple of OpenRouter-only request fields.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter").lower()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "http://localhost:8501")
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "rag-doc-search")

# Gemini via its OpenAI-compatible endpoint.
# Source: https://ai.google.dev/gemini-api/docs/openai
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")


def get_llm_settings():
    """Resolve (provider, api_key, base_url, model) from current config values.

    A function, not module-level constants, so tests (and any runtime
    monkeypatching of e.g. OPENROUTER_API_KEY) are reflected immediately --
    module-level constants would freeze whatever LLM_PROVIDER was at import
    time and silently ignore later changes.
    """
    if LLM_PROVIDER == "gemini":
        return "gemini", GEMINI_API_KEY, GEMINI_BASE_URL, GEMINI_MODEL
    return "openrouter", OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL

# --- Summarization ---
SUMMARY_LENGTHS = {
    "short": 50,
    "medium": 120,
    "long": 250,
}
