from .llm import get_llm
from .vectorstore import VectorStore
from langchain.chains import RetrievalQA

class RAGChain:
    def __init__(self):
        self.llm = get_llm()
        self.vectorstore = VectorStore()

    def load_or_build(self, path="faiss_index", text=None):
        """
        Try to load a prebuilt index at `path/`. If missing, build it from `text`.
        """
        try:
            self.vectorstore.load_local(path=path)
        except FileNotFoundError:
            if not text:
                raise
            # build & persist
            self.vectorstore.build_local(text)
            self.vectorstore.save_local(path=path)

    def build_chain(self):
        """
        Create a RetrievalQA chain using the VectorStore already built/loaded.
        """
        retriever = self.vectorstore.db.as_retriever(search_kwargs={"k": 3})
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
