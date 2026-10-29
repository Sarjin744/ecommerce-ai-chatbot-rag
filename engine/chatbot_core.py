import os
from typing import Dict, Any
from .ollama_client import OllamaClient
from .intent_classifier import IntentClassifier
from .faq_engine import FAQEngine
from .recommendation_engine import RecommendationEngine
from .sentiment_engine import SentimentEngine
from .product_qa_engine import ProductQAEngine

class EcommerceChatbot:
    """
    Main E-Commerce AI Chatbot unifying all components specified in Experiment 1:
      1. Intent Identification
      2. Semantic FAQ Search via Embeddings & Cosine Similarity
      3. Product Recommendation via TF-IDF & Cosine Similarity
      4. Customer Review Sentiment Analysis
      5. Product Q&A with Local LLM (Qwen / Ollama)
    """

    def __init__(self, data_dir: str = "data", ollama_model: str = "qwen2:0.5b"):
        self.data_dir = data_dir
        self.ollama = OllamaClient(model=ollama_model)

        products_csv = os.path.join(data_dir, "products.csv")
        faqs_csv = os.path.join(data_dir, "faqs.csv")
        reviews_csv = os.path.join(data_dir, "reviews.csv")

        # Initialize sub-engines
        self.intent_classifier = IntentClassifier(self.ollama)
        self.faq_engine = FAQEngine(faqs_csv, self.ollama)
        self.recommendation_engine = RecommendationEngine(products_csv)
        self.sentiment_engine = SentimentEngine(reviews_csv, self.ollama)
        self.product_qa_engine = ProductQAEngine(products_csv, self.ollama)

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Executes complete chatbot workflow:
        User Question -> Identify User Intent -> Route to Engine -> Generate Response
        """
        # Step 1: Intent Analysis
        intent_data = self.intent_classifier.classify(query)
        intent = intent_data["intent"]

        response_payload = {
            "query": query,
            "intent": intent,
            "confidence": intent_data["confidence"],
            "intent_source": intent_data["source"],
            "ollama_active": self.ollama.is_connected,
            "data": {}
        }

        # Step 2: Route based on intent
        if intent == "faq":
            faq_res = self.faq_engine.search(query)
            response_payload["type"] = "faq"
            response_payload["text"] = faq_res["answer"]
            response_payload["data"] = faq_res

        elif intent == "product_recommendation":
            rec_res = self.recommendation_engine.recommend(query)
            response_payload["type"] = "recommendation"
            recs = rec_res["recommendations"]
            if recs:
                items_text = "\n".join([
                    f"• **{r['name']}** ({r['brand']}) - {r['price']} | Match Score: {r['similarity_score']}\n  {r['description']}"
                    for r in recs
                ])
                response_payload["text"] = f"Here are the best product recommendations based on your requirements:\n\n{items_text}"
            else:
                response_payload["text"] = "No matching products found within the given criteria."
            response_payload["data"] = rec_res

        elif intent == "sentiment_analysis":
            sent_res = self.sentiment_engine.analyze(query)
            response_payload["type"] = "sentiment"
            response_payload["text"] = sent_res["summary"]
            response_payload["data"] = sent_res

        elif intent == "product_qa":
            qa_res = self.product_qa_engine.answer(query)
            response_payload["type"] = "product_qa"
            response_payload["text"] = qa_res["answer"]
            response_payload["data"] = qa_res

        else:
            response_payload["type"] = "general"
            response_payload["text"] = "I can assist you with product recommendations, answering product questions, FAQ support, or analyzing customer reviews!"

        return response_payload
