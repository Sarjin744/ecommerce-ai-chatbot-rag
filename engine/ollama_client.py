import json
import requests
from typing import Dict, Any, Optional, List

class OllamaClient:
    """
    Client for interacting with local Ollama API (http://localhost:11434).
    Supports model generation, chat, and embeddings.
    """
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen2:0.5b"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.is_connected = self.check_connection()

    def check_connection(self) -> bool:
        """Check if local Ollama daemon is reachable."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=1.5)
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """List available local models in Ollama."""
        if not self.check_connection():
            return []
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=2.0)
            data = resp.json()
            return [m.get("name", "") for m in data.get("models", [])]
        except Exception:
            return []

    def generate(self, prompt: str, system: Optional[str] = None, model: Optional[str] = None) -> Optional[str]:
        """Generate response using Ollama generate API."""
        if not self.check_connection():
            return None

        target_model = model or self.model
        payload: Dict[str, Any] = {
            "model": target_model,
            "prompt": prompt,
            "stream": False
        }
        if system:
            payload["system"] = system

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=30.0
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("response", "").strip()
        except Exception as e:
            return None
        return None

    def get_embedding(self, text: str, model: Optional[str] = None) -> Optional[List[float]]:
        """Compute embedding vector using Ollama embeddings API."""
        if not self.check_connection():
            return None

        target_model = model or self.model
        payload = {
            "model": target_model,
            "prompt": text
        }
        try:
            resp = requests.post(
                f"{self.base_url}/api/embeddings",
                json=payload,
                timeout=10.0
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("embedding", None)
        except Exception:
            return None
        return None
