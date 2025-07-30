from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from .embedding import BedrockEmbedding

class VectorStore:
    def __init__(self):
        self.embedder = BedrockEmbedding()
        self.db = None

    def build_store(self, texts, metadatas=None):
        """
        Create a FAISS vector store from given texts and optional metadata.
        """
        docs = [Document(page_content=text, metadata=metadatas[i] if metadatas else {}) for i, text in enumerate(texts)]
        self.db = FAISS.from_documents(docs, self.embedder)

    def save_local(self, path="faiss_index"):
        """
        Save FAISS index locally.
        """
        if self.db:
            self.db.save_local(folder_path=path)

    def load_local(self, path="faiss_index"):
        """
        Load FAISS index from local path.
        """
        self.db = FAISS.load_local(folder_path=path, embeddings=self.embedder)

    def search(self, query, k=3):
        """
        Perform similarity search and return top-k documents.
        """
        if not self.db:
            raise ValueError("Vector store is not loaded or built.")
        return self.db.similarity_search(query, k=k)
