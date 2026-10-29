import re
from typing import Dict, Any, Tuple
from .ollama_client import OllamaClient

class IntentClassifier:
    """
    Identifies the user intent from incoming queries:
      - faq: common store questions (shipping, tracking, returns, cancellation, refund, warranty)
      - product_recommendation: requests to recommend, suggest, or find products matching specs/budget
      - sentiment_analysis: asking what customers think, review summary, opinions
      - product_qa: specific technical or factual questions about a product (specs, waterproof, battery, etc.)
    """

    INTENTS = ["faq", "product_recommendation", "sentiment_analysis", "product_qa"]

    def __init__(self, ollama: OllamaClient):
        self.ollama = ollama

    def classify_with_ollama(self, query: str) -> Tuple[str, float]:
        """Classify user query using Ollama Qwen model."""
        system_prompt = (
            "You are an intent classification system for an e-commerce chatbot. "
            "Categorize the user's input into strictly ONE of the following 4 categories:\n"
            "1. faq (questions about order tracking, delivery, returns, refunds, order cancellation, warranty, payment)\n"
            "2. product_recommendation (requests to find, show, suggest, or recommend products based on budget or category)\n"
            "3. sentiment_analysis (asking for customer feedback, opinions, reviews, or what people think about a product)\n"
            "4. product_qa (questions about a specific product's features, specs, whether it is waterproof, battery life, etc.)\n\n"
            "Respond ONLY with a JSON object in this exact format: {\"intent\": \"<intent_name>\", \"confidence\": 0.95}"
        )

        prompt = f"User query: \"{query}\"\nJSON response:"
        response = self.ollama.generate(prompt=prompt, system=system_prompt)
        if response:
            try:
                import json
                cleaned = response[response.find("{"):response.rfind("}")+1]
                data = json.loads(cleaned)
                intent = data.get("intent", "").strip().lower()
                confidence = float(data.get("confidence", 0.85))
                if intent in self.INTENTS:
                    return intent, confidence
            except Exception:
                pass
        return self.classify_rule_based(query)

    def classify_rule_based(self, query: str) -> Tuple[str, float]:
        """Fast, robust rule-based intent classifier with score weighting."""
        q = query.lower().strip()

        # Sentiment Analysis keywords
        sentiment_patterns = [
            r"what (do|did) (customers|people|users) (think|say)",
            r"customer (opinion|feedback|review|sentiment|rating)",
            r"(reviews?|ratings?|opinions?) (for|of|about)",
            r"how (do|is) (customers|people) feel",
            r"sentiment (analysis|score|summary)",
            r"is it worth buying",
            r"pros and cons"
        ]
        for pat in sentiment_patterns:
            if re.search(pat, q):
                return "sentiment_analysis", 0.95

        # FAQ keywords & patterns
        faq_patterns = [
            r"\b(track|tracking)\b.*(order|shipment|parcel|package)?",
            r"\b(where is my|where's my)\b.*(parcel|order|package|item)",
            r"\b(cancel|cancellation)\b.*(order|shipment|item|delivery)?",
            r"\b(return|returning|replace|replacement|refund)\b",
            r"\b(payment mode|payment method|pay with|cod|cash on delivery|upi)\b",
            r"\b(shipping (charge|cost|fee)|delivery fee|international shipping)\b",
            r"\b(warranty|guarantee|claim warranty)\b"
        ]
        for pat in faq_patterns:
            if re.search(pat, q):
                return "faq", 0.94

        # Recommendation keywords
        recommend_patterns = [
            r"\b(recommend|suggest|show me|find me|looking for|best|top|budget|under \d+|cheap)\b",
            r"\b(gaming laptop|smartphones?|laptops?|headphones?|earbuds?|smartwatch(es)?)\b.*(under|for|with|best)"
        ]
        for pat in recommend_patterns:
            if re.search(pat, q):
                # Check if it's asking a specific spec question instead
                if not re.search(r"^(is|does|can|will|has)\b", q):
                    return "product_recommendation", 0.90

        # Product Q&A keywords (Specific factual questions)
        qa_patterns = [
            r"^(is|does|can|has|what is the price of|how much is|tell me about)\b",
            r"waterproof|water resistant|ip68|ip54",
            r"how much (ram|storage|battery|megapixels?|refresh rate)",
            r"processor|chipset|graphics|gpu|cpu|screen size"
        ]
        for pat in qa_patterns:
            if re.search(pat, q):
                return "product_qa", 0.88

        # Fallback heuristic
        if any(w in q for w in ["laptop", "phone", "headphone", "watch"]):
            if any(w in q for w in ["best", "good", "recommend", "buy", "suggest", "options"]):
                return "product_recommendation", 0.75
            return "product_qa", 0.70

        return "faq", 0.50

    def classify(self, query: str) -> Dict[str, Any]:
        """Classify user query and return intent + confidence + source."""
        if self.ollama.is_connected:
            intent, conf = self.classify_with_ollama(query)
            source = f"Ollama ({self.ollama.model})"
        else:
            intent, conf = self.classify_rule_based(query)
            source = "Semantic Heuristic Rule Engine"

        return {
            "intent": intent,
            "confidence": round(conf, 2),
            "source": source
        }
