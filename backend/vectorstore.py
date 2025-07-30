from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .embedding import BedrockEmbedding
import os

class VectorStore:
    def __init__(self):
        self.embedder = BedrockEmbedding()
        self.db = None

    def build_local(self, text, chunk_size=500, chunk_overlap=100, metadata=None):
        """
        Build FAISS vector store from a single long text
        (or combine multiple texts into one big string).
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_text(text)
        docs = [Document(page_content=chunk, metadata=metadata or {}) for chunk in chunks]
        self.db = FAISS.from_documents(docs, self.embedder)

    def save_local(self, path="faiss_index"):
        """
        Save the FAISS index to disk under `path/`.
        """
        if self.db is None:
            raise ValueError("No index to save – call build_local() first.")
        os.makedirs(path, exist_ok=True)
        self.db.save_local(folder_path=path)

    def load_local(self, path="faiss_index"):
        """
        Load an existing FAISS index from `path/`.
        """
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Vector store path '{path}' not found.")
        # Enable dangerous deserialization since we trust our own index files
        self.db = FAISS.load_local(
            folder_path=path,
            embeddings=self.embedder,
            allow_dangerous_deserialization=True
        )

    def search(self, query, k=4):
        """
        Return top‑k most similar chunks to `query`.
        """
        if self.db is None:
            raise ValueError("Vector store not built or loaded.")
        return self.db.similarity_search(query, k=k)
