import pandas as pd
import numpy as np
from typing import Dict, Any, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .ollama_client import OllamaClient

class FAQEngine:
    """
    Handles FAQ matching using Embeddings and Cosine Similarity.
    Implements Semantic Search to match conceptually identical questions
    even if the exact words differ (e.g. 'Where is my parcel?' -> 'How do I track my order?').
    """

    def __init__(self, faq_file: str, ollama: OllamaClient):
        self.faq_file = faq_file
        self.ollama = ollama
        self.df = pd.read_csv(faq_file)
        self.questions = self.df["Question"].tolist()
        self.answers = self.df["Answer"].tolist()

        # Local vectorizer fallback for embeddings
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 3), analyzer="char_wb")
        self.faq_vectors = self.vectorizer.fit_transform(self.questions)

        # Precompute Ollama embeddings if connected
        self.ollama_embeddings: List[np.ndarray] = []
        if self.ollama.is_connected:
            self._precompute_ollama_embeddings()

    def _precompute_ollama_embeddings(self):
        """Precompute embeddings for FAQ questions via Ollama."""
        vectors = []
        for q in self.questions:
            emb = self.ollama.get_embedding(q)
            if emb:
                vectors.append(np.array(emb, dtype=np.float32))
            else:
                self.ollama_embeddings = []
                return
        self.ollama_embeddings = vectors

    def search(self, query: str, threshold: float = 0.25) -> Dict[str, Any]:
        """
        Search for most similar FAQ using Cosine Similarity.
        Returns:
            matched_question, answer, similarity_score, all_scores
        """
        method = "Local Subword Semantic Vectorizer"

        # Try Ollama embeddings if available
        if self.ollama.is_connected and len(self.ollama_embeddings) == len(self.questions):
            query_emb = self.ollama.get_embedding(query)
            if query_emb is not None:
                q_vec = np.array(query_emb, dtype=np.float32).reshape(1, -1)
                emb_matrix = np.array(self.ollama_embeddings)
                sims = cosine_similarity(q_vec, emb_matrix)[0]
                method = f"Ollama Embeddings ({self.ollama.model})"
            else:
                q_vec = self.vectorizer.transform([query])
                sims = cosine_similarity(q_vec, self.faq_vectors)[0]
        else:
            q_vec = self.vectorizer.transform([query])
            sims = cosine_similarity(q_vec, self.faq_vectors)[0]

        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        # Rank all questions
        ranked_indices = np.argsort(sims)[::-1]
        top_matches = [
            {
                "question": self.questions[i],
                "answer": self.answers[i],
                "similarity": round(float(sims[i]), 4)
            }
            for i in ranked_indices[:3]
        ]

        is_match = best_score >= threshold

        return {
            "is_match": is_match,
            "matched_question": self.questions[best_idx] if is_match else None,
            "answer": self.answers[best_idx] if is_match else "I'm sorry, I couldn't find a direct FAQ answering that. Please contact support at support@shopai.karunya.edu.",
            "similarity_score": round(best_score, 4),
            "method": method,
            "top_candidates": top_matches
        }
