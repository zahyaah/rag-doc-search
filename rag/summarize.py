"""LLM-based summarization of retrieved documents."""

from rag import config
from rag.llm_client import chat_completion


def summarize_documents(
    documents: list,
    length: str = "medium",
    query: str = "",
) -> str:
    """Summarize a list of retrieved documents into one coherent summary.

    `documents` items need a "text" field (as returned by HybridSearcher.search).
    `length` is one of "short" | "medium" | "long" (see config.SUMMARY_LENGTHS),
    or an int giving an explicit target word count.
    """
    target_words = config.SUMMARY_LENGTHS.get(length, length) if isinstance(length, str) else length

    combined_text = "\n\n---\n\n".join(
        f"[Document {i+1}: {d['title']}]\n{d['text']}" for i, d in enumerate(documents)
    )
    # Truncate defensively to keep prompts within the free-tier model's context.
    combined_text = combined_text[:12000]

    system_prompt = (
        "You are a precise summarization assistant. Summarize the given documents "
        "into a single coherent summary that captures their essence. Do not add "
        "information that isn't in the documents. Do not mention 'Document 1' etc. "
        "in the output -- write it as one unified summary."
    )
    user_prompt = (
        f"User query: {query}\n\n" if query else ""
    ) + (
        f"Summarize the following document(s) in approximately {target_words} words:\n\n"
        f"{combined_text}"
    )

    summary = chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=int(target_words * 2.2) + 50,
    )
    return summary
