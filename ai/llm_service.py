"""LLM Service integration layer for natural-language dataset schema generation.

Modular design supporting Google Gemini via API key stored in environment variables,
with offline rule-based heuristic fallback if no API key is provided.
"""

from __future__ import annotations
import os
import json
import re
from typing import Any, Dict, Optional


class LLMService:
    """Service to communicate with Generative AI models (e.g. Gemini) for schema drafting."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model_name = "gemini-1.5-flash"
        self._client = None
        self._init_client()

    def _init_client(self):
        """Initialize Google Gemini client if API key is present."""
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(self.model_name)
            except Exception:
                self._client = None

    @property
    def is_available(self) -> bool:
        """Returns True if live LLM client is configured."""
        return self._client is not None

    def generate_schema_json(self, user_prompt: str) -> Dict[str, Any]:
        """Convert natural language prompt into a structured JSON schema.

        If live LLM is configured, calls model; otherwise employs heuristic NLP parser.
        """
        if self.is_available:
            try:
                system_instruction = (
                    "You are an expert Data Scientist AI. Convert the user's natural language request into a "
                    "JSON dataset schema with keys: domain, ml_task, n_rows, target_column, missing_percentage, "
                    "columns (list with name, col_type, min_val, max_val, categories). "
                    "Respond ONLY with valid JSON."
                )
                response = self._client.generate_content(
                    f"{system_instruction}\n\nUser request: {user_prompt}"
                )
                text = response.text.strip()
                # Extract JSON block
                json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))
                return json.loads(text)
            except Exception:
                pass  # Fall back to heuristic parser

        return self._heuristic_parse(user_prompt)

    def _heuristic_parse(self, prompt: str) -> Dict[str, Any]:
        """Offline heuristic parser to extract domain, rows, task, and target from prompt."""
        p_lower = prompt.lower()

        # Rows extraction
        n_rows = 1000
        row_match = re.search(r"(\d+[\d,]*)\s*(?:rows|records|samples)", p_lower)
        if row_match:
            try:
                n_rows = int(row_match.group(1).replace(",", ""))
            except ValueError:
                pass

        # Domain extraction
        domain = "General ML"
        if any(w in p_lower for w in ["health", "patient", "medical", "disease", "hospital", "clinical"]):
            domain = "Healthcare"
        elif any(w in p_lower for w in ["bank", "loan", "credit", "finance", "fraud", "default"]):
            domain = "Banking and Finance"
        elif any(w in p_lower for w in ["ecommerce", "e-commerce", "churn", "shop", "customer", "purchase", "order"]):
            domain = "E-commerce"
        elif any(w in p_lower for w in ["student", "school", "exam", "education", "grade", "gpa"]):
            domain = "Education"
        elif any(w in p_lower for w in ["crop", "soil", "agriculture", "farm", "yield"]):
            domain = "Agriculture"

        # Task extraction
        ml_task = "Classification"
        if "regress" in p_lower or "predict score" in p_lower or "continuous" in p_lower:
            ml_task = "Regression"
        elif "cluster" in p_lower or "unsupervised" in p_lower or "grouping" in p_lower:
            ml_task = "Clustering"

        # Missing percentage
        missing_pct = 5.0
        missing_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:missing|null)", p_lower)
        if missing_match:
            try:
                missing_pct = float(missing_match.group(1))
            except ValueError:
                pass

        return {
            "domain": domain,
            "ml_task": ml_task,
            "n_rows": n_rows,
            "missing_percentage": missing_pct,
            "source": "heuristic_nlp_parser",
        }
