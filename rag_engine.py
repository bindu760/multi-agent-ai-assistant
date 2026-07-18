"""
rag_engine.py
--------------
Yesma euta simple, dependency-light RAG (Retrieval Augmented Generation) engine
banaeko chu. Kina TF-IDF use gareko bhaneko:

  - sentence-transformers/HuggingFace embedding models use garda internet bata
    model weights download garnu parxa (kahile kahi slow ya blocked huna sakxa).
  - TF-IDF (scikit-learn) chai pure offline ho, instant load huncha, ra college/
    career jasto domain-specific short documents ko lagi ramrai sanga kaam garxa.

  Later, chaheko bhane yo class lai Chroma + sentence-transformers embedding
  ma pani easily swap garna milxa (interface same rakheko chu: .query(question)).
"""

import os
import glob
from dataclasses import dataclass
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    source: str
    text: str


class SimpleRAG:
    """A tiny TF-IDF based retriever over a folder of .txt files."""

    def __init__(self, data_dir: str, chunk_size: int = 180, chunk_overlap: int = 40):
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks: List[Chunk] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        self._build_index()

    # ---------- Indexing ----------
    def _load_documents(self):
        docs = []
        pattern = os.path.join(self.data_dir, "**", "*.txt")
        for path in glob.glob(pattern, recursive=True):
            with open(path, "r", encoding="utf-8") as f:
                docs.append((os.path.basename(path), f.read()))
        return docs

    def _chunk_text(self, text: str) -> List[str]:
        words = text.split()
        if not words:
            return []
        chunks = []
        step = max(self.chunk_size - self.chunk_overlap, 1)
        for i in range(0, len(words), step):
            piece = " ".join(words[i:i + self.chunk_size])
            if piece.strip():
                chunks.append(piece)
            if i + self.chunk_size >= len(words):
                break
        return chunks

    def _build_index(self):
        documents = self._load_documents()
        for fname, text in documents:
            for piece in self._chunk_text(text):
                self.chunks.append(Chunk(source=fname, text=piece))

        if not self.chunks:
            # Fallback so vectorizer never crashes on an empty corpus
            self.chunks.append(Chunk(source="empty", text="No knowledge base documents found."))

        corpus = [c.text for c in self.chunks]
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
        self.matrix = self.vectorizer.fit_transform(corpus)

    def reload(self):
        """Call this if you add/update .txt files in data_dir at runtime."""
        self.chunks = []
        self._build_index()

    # ---------- Querying ----------
    def query(self, question: str, top_k: int = 4) -> str:
        """Returns a formatted string of the top_k most relevant chunks."""
        if self.vectorizer is None or self.matrix is None:
            return "Knowledge base not initialized."

        q_vec = self.vectorizer.transform([question])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        top_idx = sims.argsort()[::-1][:top_k]

        results = []
        for idx in top_idx:
            if sims[idx] <= 0:
                continue
            chunk = self.chunks[idx]
            results.append(f"[Source: {chunk.source} | relevance: {sims[idx]:.2f}]\n{chunk.text}")

        if not results:
            return "No relevant information found in this knowledge base for the query."

        return "\n\n---\n\n".join(results)


# ---------------------------------------------------------------------------
# Two separate knowledge bases -> two separate RAG tools for the agent
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COLLEGE_DATA_DIR = os.path.join(BASE_DIR, "data", "college_admission")
CAREER_DATA_DIR = os.path.join(BASE_DIR, "data", "it_careers")

_college_rag_instance = None
_career_rag_instance = None


def get_college_rag() -> SimpleRAG:
    global _college_rag_instance
    if _college_rag_instance is None:
        _college_rag_instance = SimpleRAG(COLLEGE_DATA_DIR)
    return _college_rag_instance


def get_career_rag() -> SimpleRAG:
    global _career_rag_instance
    if _career_rag_instance is None:
        _career_rag_instance = SimpleRAG(CAREER_DATA_DIR)
    return _career_rag_instance