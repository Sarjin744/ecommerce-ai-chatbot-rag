import re
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class RecommendationEngine:
    """
    Recommends products based on user preferences using TF-IDF and Cosine Similarity.
    Also extracts price constraints (e.g. 'under 50000', 'below 100000').
    """

    def __init__(self, products_file: str):
        self.products_file = products_file
        self.df = pd.read_csv(products_file)

        # Build composite text representation for TF-IDF indexing
        self.df["content"] = (
            self.df["Product Name"] + " " +
            self.df["Brand"] + " " +
            self.df["Category"] + " " +
            self.df["Description"]
        )

        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["content"])

    def extract_budget(self, query: str) -> Optional[float]:
        """Extract budget limit from queries like 'under 30000', 'below 50k', '< 100000'."""
        q = query.lower()
        # Match k suffix like 50k
        k_match = re.search(r"(?:under|below|less than|max)\s*(\d+(?:\.\d+)?)\s*k\b", q)
        if k_match:
            return float(k_match.group(1)) * 1000

        # Match direct numeric amount
        num_match = re.search(r"(?:under|below|less than|within|budget of)\s*(?:rs\.?|inr)?\s*(\d[\d,]*)", q)
        if num_match:
            clean_num = num_match.group(1).replace(",", "")
            return float(clean_num)

        return None

    def recommend(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Calculates TF-IDF Cosine Similarity for the query against product descriptions.
        Ranks and returns top matching products.
        """
        budget = self.extract_budget(query)
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        # Extract top important TF-IDF keywords from the query
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        query_tfidf_indices = query_vec.toarray()[0].argsort()[::-1]
        top_keywords = [
            feature_names[i]
            for i in query_tfidf_indices
            if query_vec[0, i] > 0
        ][:6]

        ranked_indices = np.argsort(sims)[::-1]
        results: List[Dict[str, Any]] = []

        for idx in ranked_indices:
            score = float(sims[idx])
            prod = self.df.iloc[idx].to_dict()
            price = float(prod.get("Price", 0))

            # Apply budget constraint if specified
            if budget is not None and price > budget:
                continue

            # Only include relevant items
            if score > 0.05 or len(results) == 0:
                results.append({
                    "product_id": prod.get("Product ID"),
                    "name": prod.get("Product Name"),
                    "brand": prod.get("Brand"),
                    "category": prod.get("Category"),
                    "price": f"Rs. {price:,.2f}",
                    "description": prod.get("Description"),
                    "similarity_score": round(score, 4)
                })

            if len(results) >= top_k:
                break

        return {
            "query": query,
            "budget_filter": f"<= Rs. {budget:,.2f}" if budget else "None",
            "extracted_tfidf_keywords": top_keywords,
            "total_matches": len(results),
            "recommendations": results
        }
