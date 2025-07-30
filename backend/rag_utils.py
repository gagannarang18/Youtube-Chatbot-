def format_answer(result):
    """
    Format the output of the RAG chain for display.
    """
    answer = result['result']
    sources = result.get('source_documents', [])

    formatted_sources = ""
    for i, doc in enumerate(sources, 1):
        source_info = doc.metadata if doc.metadata else {}
        formatted_sources += f"\n\n**Source {i}:**\n{doc.page_content.strip()[:500]}..."
        if source_info:
            formatted_sources += f"\n\n_Metadata: {source_info}_"

    return f"### Answer:\n{answer}\n\n---\n### Sources:{formatted_sources}"
def format_answer(result):
    """
    Format the output of the RAG chain for display.
    """
    answer = result['result']
    sources = result.get('source_documents', [])

    formatted_sources = ""
    for i, doc in enumerate(sources, 1):
        source_info = doc.metadata if doc.metadata else {}
        formatted_sources += f"\n\n**Source {i}:**\n{doc.page_content.strip()[:500]}..."
        if source_info:
            formatted_sources += f"\n\n_Metadata: {source_info}_"

    return f"### Answer:\n{answer}\n\n---\n### Sources:{formatted_sources}"
