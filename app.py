"""Streamlit UI for the RAG document search + summarization system.

Run: uv run streamlit run app.py
"""

import streamlit as st

from rag import config
from rag.data_prep import load_jsonl
from rag.llm_client import LLMError
from rag.search import HybridSearcher
from rag.suggest import suggest_queries
from rag.summarize import summarize_documents

st.set_page_config(page_title="RAG Document Search", layout="wide")

PAGE_SIZE = 5


@st.cache_resource
def get_searcher() -> HybridSearcher:
    return HybridSearcher()


@st.cache_data
def get_titles() -> list:
    docs = load_jsonl(config.CORPUS_PATH)
    return [d["title"] for d in docs]


def main():
    st.title("Document Search & Summarization")
    st.caption(
        f"Hybrid BM25 + embedding search over {len(get_titles())} CNN/DailyMail articles, "
        "summarized on demand via an LLM."
    )

    if not config.CORPUS_PATH.exists():
        st.error(
            f"Corpus not found at {config.CORPUS_PATH}. Run `uv run python -m rag.data_prep` "
            "and `uv run python -m rag.indexing` first."
        )
        return

    titles = get_titles()

    query = st.text_input("Search query", placeholder="e.g. climate summit agreement")

    if query.strip():
        suggestions = suggest_queries(query, titles, max_suggestions=5)
        if suggestions and query not in titles:
            with st.expander("Suggestions", expanded=False):
                for s in suggestions:
                    st.write(f"- {s}")

    col1, col2, col3 = st.columns(3)
    with col1:
        top_n = st.slider("Number of results", min_value=1, max_value=20, value=10)
    with col2:
        summary_length = st.selectbox("Summary length", ["short", "medium", "long"], index=1)
    with col3:
        page = st.number_input("Page", min_value=1, value=1, step=1)

    if not query.strip():
        st.info("Enter a query above to search the corpus.")
        return

    searcher = get_searcher()
    with st.spinner("Searching..."):
        results = searcher.search(query, top_n=top_n)

    if not results:
        st.warning("No results found.")
        return

    start = (page - 1) * PAGE_SIZE
    end = start + PAGE_SIZE
    page_results = results[start:end]
    total_pages = max(1, (len(results) + PAGE_SIZE - 1) // PAGE_SIZE)

    st.subheader(f"Results {start + 1}-{min(end, len(results))} of {len(results)} (page {page}/{total_pages})")

    selected_for_summary = []
    for r in page_results:
        with st.container(border=True):
            checked = st.checkbox(
                f"**#{r['rank']} — {r['title']}**  ·  score {r['score']:.3f} "
                f"(bm25 {r['bm25_score']:.2f} / embed {r['embedding_score']:.2f})",
                key=f"select_{r['doc_id']}",
            )
            st.write(r["text"][:400] + ("..." if len(r["text"]) > 400 else ""))
            if checked:
                selected_for_summary.append(r)

    st.divider()
    n_to_summarize = len(selected_for_summary) or len(page_results)
    label = (
        f"Summarize {len(selected_for_summary)} selected document(s)"
        if selected_for_summary
        else f"Summarize top {len(page_results)} result(s) on this page"
    )
    if st.button(label, type="primary"):
        docs_to_summarize = selected_for_summary or page_results
        with st.spinner("Summarizing with LLM..."):
            try:
                summary = summarize_documents(docs_to_summarize, length=summary_length, query=query)
                st.subheader("Summary")
                st.write(summary)
            except LLMError as e:
                st.error(str(e))


if __name__ == "__main__":
    main()
