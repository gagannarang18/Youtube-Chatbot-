from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .embedding import BedrockEmbedding  # Note: Make sure filename is plural
import os

class VectorStore:
    def __init__(self):
        self.embedder = BedrockEmbedding()
        self.db = None

    def build_store(self, text, chunk_size=500, chunk_overlap=100, metadata=None):
        """
        Build FAISS vector store from a single long text.
        Automatically splits into chunks.
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_text(text)
        docs = [Document(page_content=chunk, metadata=metadata or {}) for chunk in chunks]
        self.db = FAISS.from_documents(docs, self.embedder)

    def save_local(self, path="faiss_index"):
        """
        Save the FAISS index locally to the given path.
        """
        if self.db:
            self.db.save_local(folder_path=path)

    def load_local(self, path="faiss_index"):
        """
        Load FAISS index from the given path.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vector store path '{path}' not found.")
        self.db = FAISS.load_local(folder_path=path, embeddings=self.embedder)

    def search(self, query, k=4):
        """
        Perform similarity search and return top-k chunks.
        """
        if not self.db:
            raise ValueError("Vector store is not built or loaded.")
        return self.db.similarity_search(query, k=k)
