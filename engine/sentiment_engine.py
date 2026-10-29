import re
import pandas as pd
from typing import Dict, Any, List
from .ollama_client import OllamaClient

class SentimentEngine:
    """
    Analyzes customer reviews to determine sentiment (Positive, Negative, Neutral).
    Computes overall sentiment distribution and summarizes customer opinions.
    """

    POSITIVE_WORDS = {
        "excellent", "great", "good", "amazing", "love", "loved", "super", "best",
        "phenomenal", "breathtaking", "crisp", "smooth", "silky", "fast", "mindblowing",
        "silent", "gorgeous", "blazing", "stunning", "saved", "waterproof", "vibrant",
        "leading", "rich", "comfortable", "clarity", "accurate", "bright", "seamless",
        "reliable", "satisfying", "worth", "durable", "clear", "high"
    }

    NEGATIVE_WORDS = {
        "poor", "bad", "terrible", "worst", "noise", "noisy", "heating", "issue", "issues",
        "hot", "bulky", "heavy", "drops", "drop", "expensive", "pricy", "pricey",
        "grainy", "bloatware", "annoying", "cannot", "cant", "delay", "defective",
        "weak", "slow", "plastic", "low", "drains"
    }

    def __init__(self, reviews_file: str, ollama: OllamaClient):
        self.reviews_file = reviews_file
        self.ollama = ollama
        self.df = pd.read_csv(reviews_file)

    def classify_review_sentiment(self, text: str) -> str:
        """Classify single review text into Positive, Negative, or Neutral."""
        # Clean text
        words = re.findall(r"\b\w+\b", text.lower())
        pos_count = sum(1 for w in words if w in self.POSITIVE_WORDS)
        neg_count = sum(1 for w in words if w in self.NEGATIVE_WORDS)

        if pos_count > neg_count:
            return "Positive"
        elif neg_count > pos_count:
            return "Negative"
        else:
            return "Neutral"

    def match_product(self, query: str) -> str:
        """Find the most matching product or category from the user query."""
        q = query.lower()
        all_products = self.df["Product"].unique().tolist()

        for prod in all_products:
            if prod.lower() in q:
                return prod

        # Check keyword matches with word boundaries or specific order
        if "headphone" in q or "sony" in q or "audio" in q or "earphone" in q:
            return "Sony WH-1000XM5 Wireless Headphones"
        if "watch" in q:
            return "Samsung Galaxy Watch 6"
        if "laptop" in q or "predator" in q or "rog" in q:
            return "Predator Helios 16 Gaming Laptop"
        if "iphone" in q or "apple phone" in q:
            return "iPhone 15 Pro"
        if "samsung" in q or "galaxy" in q or "s24" in q:
            return "Galaxy S24 Ultra"
        if "redmi" in q or "xiaomi" in q or re.search(r"\bphones?\b", q):
            return "Redmi Note 13 Pro"

        return all_products[0]

    def analyze(self, query: str) -> Dict[str, Any]:
        """
        Filters reviews for the specified product and performs sentiment analysis.
        """
        target_product = self.match_product(query)
        matching_reviews = self.df[self.df["Product"].str.lower().str.contains(target_product.lower()[:8])]

        if matching_reviews.empty:
            matching_reviews = self.df.head(4)
            target_product = "General Products"

        analyzed_items: List[Dict[str, str]] = []
        pos, neg, neu = 0, 0, 0

        for _, row in matching_reviews.iterrows():
            rev = str(row["Review"])
            sentiment = self.classify_review_sentiment(rev)
            if sentiment == "Positive":
                pos += 1
            elif sentiment == "Negative":
                neg += 1
            else:
                neu += 1

            analyzed_items.append({
                "product": str(row["Product"]),
                "review": rev,
                "sentiment": sentiment
            })

        total = len(analyzed_items)
        pos_pct = round((pos / total) * 100, 1) if total else 0
        neg_pct = round((neg / total) * 100, 1) if total else 0
        neu_pct = round((neu / total) * 100, 1) if total else 0

        if pos_pct >= 60:
            consensus = "Mostly Positive 👍"
        elif neg_pct >= 50:
            consensus = "Mostly Negative 👎"
        else:
            consensus = "Mixed / Neutral ⚖️"

        summary_text = (
            f"Based on {total} customer reviews for **{target_product}**:\n"
            f"- **Positive Sentiment:** {pos_pct}% ({pos}/{total})\n"
            f"- **Negative Sentiment:** {neg_pct}% ({neg}/{total})\n"
            f"- **Neutral Sentiment:** {neu_pct}% ({neu}/{total})\n"
            f"**Overall Customer Consensus:** {consensus}"
        )

        return {
            "product": target_product,
            "total_reviews": total,
            "counts": {"Positive": pos, "Negative": neg, "Neutral": neu},
            "percentages": {"Positive": pos_pct, "Negative": neg_pct, "Neutral": neu_pct},
            "consensus": consensus,
            "summary": summary_text,
            "reviews": analyzed_items
        }
