from langchain.chains import RetrievalQA
from .llm import get_llm
from .vectorstore import VectorStore

class RAGChain:
    def __init__(self):
        self.llm = get_llm()
        self.vectorstore = VectorStore()

    def load_vectorstore(self, path="faiss_index"):
        """
        Load pre-built FAISS vector store.
        """
        self.vectorstore.load_local(path=path)

    def build_chain(self):
        """
        Create a RetrievalQA chain using the loaded vector store.
        """
        retriever = self.vectorstore.db.as_retriever(search_kwargs={"k": 3})
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # can switch to "map_reduce" or "refine" if needed
            retriever=retriever,
            return_source_documents=True
        )
        return chain
