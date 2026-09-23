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

    # Reset to page 1 whenever the query changes -- otherwise a stale page
    # number from a previous, longer result set can point past the end of a
    # new, shorter one and silently render nothing.
    if st.session_state.get("_last_query") != query:
        st.session_state["_last_query"] = query
        st.session_state["page"] = 1
    st.session_state.setdefault("page", 1)

    if query.strip():
        suggestions = suggest_queries(query, titles, max_suggestions=5)
        if suggestions and query not in titles:
            with st.expander("Suggestions", expanded=False):
                for s in suggestions:
                    st.write(f"- {s}")

    col1, col2 = st.columns(2)
    with col1:
        top_n = st.slider("Number of results", min_value=1, max_value=20, value=10)
    with col2:
        summary_length = st.selectbox("Summary length", ["short", "medium", "long"], index=1)

    if not query.strip():
        st.info("Enter a query above to search the corpus.")
        return

    searcher = get_searcher()
    with st.spinner("Searching..."):
        results = searcher.search(query, top_n=top_n)

    if not results:
        st.warning("No results found.")
        return

    total_pages = max(1, (len(results) + PAGE_SIZE - 1) // PAGE_SIZE)
    # Clamp in case top_n shrank the result count below the stored page.
    st.session_state["page"] = min(st.session_state["page"], total_pages)
    page = st.session_state["page"]

    start = (page - 1) * PAGE_SIZE
    end = start + PAGE_SIZE
    page_results = results[start:end]

    nav_prev, nav_label, nav_next = st.columns([1, 3, 1])
    with nav_prev:
        if st.button("Previous", disabled=(page <= 1), use_container_width=True):
            st.session_state["page"] -= 1
            st.rerun()
    with nav_label:
        st.markdown(
            f"<div style='text-align:center; padding-top:0.4em'>"
            f"Results {start + 1}-{min(end, len(results))} of {len(results)} "
            f"&nbsp;&middot;&nbsp; Page {page}/{total_pages}</div>",
            unsafe_allow_html=True,
        )
    with nav_next:
        if st.button("Next", disabled=(page >= total_pages), use_container_width=True):
            st.session_state["page"] += 1
            st.rerun()

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
                with st.container(border=True):
                    st.markdown(summary)
                st.download_button(
                    "Download as Markdown (.md)",
                    data=summary,
                    file_name="summary.md",
                    mime="text/markdown",
                )
            except LLMError as e:
                st.error(str(e))


if __name__ == "__main__":
    main()
