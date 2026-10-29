import re
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .ollama_client import OllamaClient

class ProductQAEngine:
    """
    Answers technical and factual questions about products
    (e.g., 'Is this phone waterproof?', 'How much RAM does the Predator laptop have?').
    Combines context retrieval from products.csv with Ollama Qwen generation.
    """

    def __init__(self, products_file: str, ollama: OllamaClient):
        self.products_file = products_file
        self.ollama = ollama
        self.df = pd.read_csv(products_file)

        # Build retrieval index
        self.df["search_text"] = (
            self.df["Product Name"] + " " +
            self.df["Brand"] + " " +
            self.df["Category"] + " " +
            self.df["Description"]
        )
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.df["search_text"])

    def retrieve_product_context(self, query: str) -> Dict[str, Any]:
        """Find most relevant product record for the query."""
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        best_idx = int(sims.argmax())
        prod = self.df.iloc[best_idx].to_dict()
        return {
            "product": prod,
            "similarity": float(sims[best_idx])
        }

    def answer_with_ollama(self, query: str, product: Dict[str, Any]) -> Optional[str]:
        """Send context + question to local Ollama Qwen model."""
        system_prompt = (
            "You are a helpful and concise e-commerce AI shopping assistant. "
            "Answer the customer's question directly based ONLY on the provided product specifications. "
            "Keep the answer concise (2-4 sentences max)."
        )

        context = (
            f"Product Name: {product.get('Product Name')}\n"
            f"Brand: {product.get('Brand')}\n"
            f"Category: {product.get('Category')}\n"
            f"Price: Rs. {product.get('Price')}\n"
            f"Specifications & Description: {product.get('Description')}"
        )

        prompt = f"Product Information:\n{context}\n\nCustomer Question: {query}\n\nAnswer:"
        return self.ollama.generate(prompt=prompt, system=system_prompt)

    def answer_extractive_fallback(self, query: str, product: Dict[str, Any]) -> str:
        """Rule-based extractor for when Ollama is offline or warming up."""
        name = product.get("Product Name", "Product")
        desc = product.get("Description", "")
        q = query.lower()

        # Waterproof question
        if "waterproof" in q or "water" in q or "ip68" in q or "ip54" in q:
            if "ip68" in desc.lower():
                return f"Yes, the {name} features an IP68 water and dust resistance rating. {desc}"
            elif "ip54" in desc.lower():
                return f"The {name} comes with an IP54 splash resistance rating, meaning it can withstand light splashes of water, but is not fully waterproof."
            else:
                return f"Regarding water resistance for {name}: {desc}"

        # RAM / Storage / Battery / Processor questions
        if "ram" in q or "memory" in q:
            ram_match = re.search(r"\b(\d+GB\s*(?:DDR\d|unified\s*memory|RAM))\b", desc, re.IGNORECASE)
            if ram_match:
                return f"The {name} is equipped with {ram_match.group(1)}."
        if "battery" in q:
            battery_match = re.search(r"(\d+\s*hours?|\d+mAh)", desc, re.IGNORECASE)
            if battery_match:
                return f"The {name} offers {battery_match.group(1)} battery performance."

        return f"Regarding '{name}': {desc} It is priced at Rs. {product.get('Price'):,}."

    def answer(self, query: str) -> Dict[str, Any]:
        """Orchestrates product retrieval and QA generation."""
        retrieval = self.retrieve_product_context(query)
        prod = retrieval["product"]

        if self.ollama.is_connected:
            generated = self.answer_with_ollama(query, prod)
            if generated:
                return {
                    "matched_product": prod["Product Name"],
                    "brand": prod["Brand"],
                    "price": f"Rs. {prod['Price']:,}",
                    "answer": generated,
                    "engine": f"Ollama Local Qwen ({self.ollama.model})"
                }

        # Fallback to extractive
        fallback_ans = self.answer_extractive_fallback(query, prod)
        return {
            "matched_product": prod["Product Name"],
            "brand": prod["Brand"],
            "price": f"Rs. {prod['Price']:,}",
            "answer": fallback_ans,
            "engine": "Context-Aware Specification Extractor (Fallback)"
        }
