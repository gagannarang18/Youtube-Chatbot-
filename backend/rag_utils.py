def format_answer(result):
    """
    Turn the raw RetrievalQA output into a markdown‑friendly string.
    """
    answer = result["result"]
    sources = result.get("source_documents", [])

    md = f"### Answer:\n{answer}\n\n---\n### Sources:"
    for i, doc in enumerate(sources, 1):
        snippet = doc.page_content.strip().replace("\n", " ")[:500]
        meta = doc.metadata or {}
        md += (
            f"\n\n**Source {i}:**\n"
            f"{snippet}..."
            + (f"\n\n_Metadata: {meta}_" if meta else "")
        )
    return md
