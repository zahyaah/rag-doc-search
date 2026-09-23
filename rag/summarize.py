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
        "in the output -- write it as one unified summary.\n\n"
        "Format the output as Markdown, exactly like this:\n"
        "**TL;DR:** one sentence capturing the single most important point.\n\n"
        "**Key points:**\n"
        "- first key fact\n"
        "- second key fact\n"
        "- (3-6 bullets total, each one sentence, ordered by importance)\n\n"
        "Do not add any other section, heading, or preamble beyond TL;DR and Key points.\n"
        f"Stay within {target_words} words total (TL;DR + all bullets combined) -- "
        "this is a hard limit, not a suggestion. Prioritize the most important facts "
        "over completeness if the two conflict."
    )
    user_prompt = (
        f"User query: {query}\n\n" if query else ""
    ) + (
        f"Summarize the following document(s) in {target_words} words or fewer:\n\n"
        f"{combined_text}"
    )

    summary = chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        # Tighter cap than before (was 2.2x + 50): the wider budget let the
        # model habitually overshoot the requested word count by 11-86% in
        # eval (see REPORT.md limitation). ~1.5 words/token plus a small
        # fixed allowance for markdown syntax (**, -, headers) is enough for
        # the target word count without leaving room to ramble.
        max_tokens=int(target_words * 1.5) + 40,
    )
    return summary
