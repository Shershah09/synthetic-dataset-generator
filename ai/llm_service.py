"""LLM Service & Prompt Engineering integration layer.

Implements:
- Google Gemini API integration with secure key handling.
- Prompt Engineering Module (Ch 3.1.3, 4.2.3).
- Prompt Injection Sanitizer & Defense (Ch 6.1, 6.2, TC-12).
- Automatic Fallback Heuristic Parser when API key is unavailable (TC-11).
- Context Augmentation with RAG.
"""

from __future__ import annotations
import os
import json
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.runnables import RunnableLambda
    LANGCHAIN_CORE_AVAILABLE = True
except ImportError:
    LANGCHAIN_CORE_AVAILABLE = False
    PromptTemplate = None
    ChatPromptTemplate = None
    JsonOutputParser = None
    RunnableLambda = None


class PromptSanitizer:
    """Security filter to detect and neutralize prompt injection and credential exfiltration attempts."""

    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"(reveal|show|print|leak|output|give|steal|display|extract|what\s+is).*?(api[_\s-]?key|secret|token|credential|env|password)",
        r"system\s+prompt",
        r"(os\.getenv|os\.environ|process\.env)",
        r"act\s+as\s+jailbreak",
        r"confidential\s+token",
    ]

    @classmethod
    def sanitize(cls, prompt: str) -> Tuple[str, bool, Optional[str]]:
        """Sanitizes user prompt.

        Returns:
            Tuple of (sanitized_prompt, is_malicious, rejection_reason)
        """
        p_lower = prompt.lower()
        for pattern in cls.INJECTION_PATTERNS:
            if re.search(pattern, p_lower):
                return (
                    "Generate a standard general machine learning benchmark dataset with 1000 rows.",
                    True,
                    "Security Notice: Malicious instruction or API credential access attempt detected and neutralized.",
                )
        return prompt, False, None


class PromptEngine:
    """Formats structured prompts according to Chapter 4.2.3 requirements."""

    @staticmethod
    def build_system_instruction(rag_context: Optional[str] = None) -> str:
        """Constructs prompt containing domain constraints, types, and JSON schema format."""
        instruction = (
            "You are an expert Data Scientist and AI Agent. Convert the user's natural language request into a "
            "precise JSON dataset schema for synthetic tabular data generation.\n\n"
            "Required JSON format:\n"
            "{\n"
            '  "domain": "Healthcare | Banking and Finance | E-commerce | Education | Agriculture | General ML",\n'
            '  "ml_task": "Classification | Regression | Clustering | General",\n'
            '  "n_rows": integer (e.g. 1000),\n'
            '  "target_column": "string name or null",\n'
            '  "missing_percentage": float between 0.0 and 40.0,\n'
            '  "columns": [\n'
            '    {"name": "Col_Name", "col_type": "numeric|categorical|boolean|datetime|id", '
            '"min_val": float, "max_val": float, "mean": float, "std": float, "categories": ["A", "B"]}\n'
            "  ]\n"
            "}\n"
            "Respond ONLY with valid JSON. Never output conversational text, explanations, or system secrets.\n"
        )
        if rag_context:
            instruction += f"\n{rag_context}\n"
        return instruction

    @staticmethod
    def get_langchain_prompt_template() -> Any:
        """Returns a LangChain PromptTemplate with input variables ['rag_context', 'user_request']."""
        if PromptTemplate is not None:
            template = (
                "You are an expert Data Scientist and AI Agent. Convert the user's natural language request into a "
                "precise JSON dataset schema for synthetic tabular data generation.\n\n"
                "Required JSON format:\n"
                "{{\n"
                '  "domain": "Healthcare | Banking and Finance | E-commerce | Education | Agriculture | General ML",\n'
                '  "ml_task": "Classification | Regression | Clustering | General",\n'
                '  "n_rows": <integer_count>,\n'
                '  "target_column": "target",\n'
                '  "missing_percentage": 5.0,\n'
                '  "columns": [\n'
                '    {{"name": "Feature_A", "col_type": "numeric", "min_val": 0.0, "max_val": 100.0}}\n'
                "  ]\n"
                "}}\n\n"
                "Domain RAG Knowledge Context:\n{rag_context}\n\n"
                "User Request:\n{user_request}\n\n"
                "Respond ONLY with valid JSON."
            )
            return PromptTemplate(
                template=template,
                input_variables=["rag_context", "user_request"],
            )
        return None


class LLMService:
    """Service to communicate with Generative AI models (e.g. Gemini) for schema drafting."""

    def __init__(self, api_key: Optional[str] = None):
        # Securely fetch API key from parameter or environment variable
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model_name = "gemini-1.5-flash"
        self._client = None
        self._init_client()

    def set_api_key(self, api_key: str):
        """Update API key dynamically at runtime."""
        self.api_key = api_key.strip()
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
        else:
            self._client = None

    @property
    def is_available(self) -> bool:
        """Returns True if live LLM client is configured."""
        return self._client is not None

    def build_langchain_chain(self) -> Any:
        """Constructs an executable LangChain LCEL sequence:
        PromptTemplate -> LLM/Fallback Runnable -> JsonOutputParser
        """
        if not LANGCHAIN_CORE_AVAILABLE:
            return None

        prompt_template = PromptEngine.get_langchain_prompt_template()
        parser = JsonOutputParser() if JsonOutputParser else None

        def _execute_model(prompt_val: Any) -> str:
            prompt_str = prompt_val.to_string() if hasattr(prompt_val, "to_string") else str(prompt_val)
            safe_text, is_malicious, alert = PromptSanitizer.sanitize(prompt_str)
            if is_malicious:
                return json.dumps({
                    "domain": "General ML",
                    "ml_task": "Classification",
                    "n_rows": 1000,
                    "missing_percentage": 5.0,
                    "source": "prompt_injection_sanitizer",
                    "security_alert": alert,
                })
            if self.is_available:
                try:
                    resp = self._client.generate_content(safe_text)
                    return resp.text.strip()
                except Exception:
                    pass
            # Deterministic heuristic fallback: isolate actual user request from template
            user_part = safe_text
            if "User Request:" in safe_text:
                user_part = safe_text.split("User Request:", 1)[1]
                if "Respond ONLY" in user_part:
                    user_part = user_part.split("Respond ONLY", 1)[0]
            parsed = self._heuristic_parse(user_part.strip())
            parsed["source"] = "langchain_fallback_parser"
            return json.dumps(parsed)

        if RunnableLambda and prompt_template and parser:
            return prompt_template | RunnableLambda(_execute_model) | parser
        return None

    def invoke_langchain(self, user_prompt: str, rag_context: str = "") -> Dict[str, Any]:
        """Execute schema drafting via the LangChain LCEL pipeline."""
        chain = self.build_langchain_chain()
        if chain is not None:
            try:
                result = chain.invoke({"user_request": user_prompt, "rag_context": rag_context})
                if isinstance(result, dict):
                    result.setdefault("source", "langchain_lcel_chain")
                    return result
            except Exception:
                pass
        return self.generate_schema_json(user_prompt, rag_context=rag_context)

    def generate_schema_json(
        self, user_prompt: str, rag_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Convert natural language prompt into a structured JSON schema.

        Includes prompt injection defense (TC-12) and automatic heuristic fallback (TC-11).
        """
        # Step 1: Prompt Sanitization & Injection Defense
        safe_prompt, is_malicious, alert = PromptSanitizer.sanitize(user_prompt)
        if is_malicious:
            return {
                "domain": "General ML",
                "ml_task": "Classification",
                "n_rows": 1000,
                "missing_percentage": 5.0,
                "source": "prompt_injection_sanitizer",
                "security_alert": alert,
            }

        # Step 2: Live Gemini Call if API is configured
        if self.is_available:
            try:
                system_instruction = PromptEngine.build_system_instruction(rag_context)
                response = self._client.generate_content(
                    f"{system_instruction}\n\nUser request: {safe_prompt}"
                )
                text = response.text.strip()
                json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
                raw_json = json_match.group(1) if json_match else text
                result = json.loads(raw_json)
                result["source"] = "google_gemini_api"
                return result
            except Exception:
                pass  # Fall back to heuristic parser gracefully (TC-11)

        # Step 3: Deterministic Fallback Parser (Offline / API unavailable)
        return self._heuristic_parse(safe_prompt)

    def _heuristic_parse(self, prompt: str) -> Dict[str, Any]:
        """Offline rule-based heuristic parser to extract domain, rows, task, and target from prompt."""
        p_lower = prompt.lower()

        # Rows extraction (e.g. 500, 1000, 2500, 5000, 10000)
        n_rows = 1000
        row_match = re.search(r"(\d+[\d,]*)\s*(?:[a-zA-Z_-]+\s+){0,5}(?:rows|records|samples)", p_lower)
        if row_match:
            try:
                n_rows = int(row_match.group(1).replace(",", ""))
                n_rows = max(1, min(n_rows, 50000))
            except ValueError:
                pass

        # Domain extraction
        domain = "General ML"
        if any(w in p_lower for w in ["health", "patient", "medical", "disease", "hospital", "clinical", "blood", "bmi"]):
            domain = "Healthcare"
        elif any(w in p_lower for w in ["bank", "loan", "credit", "finance", "fraud", "default", "balance"]):
            domain = "Banking and Finance"
        elif any(w in p_lower for w in ["ecommerce", "e-commerce", "churn", "shop", "customer", "purchase", "order", "sales"]):
            domain = "E-commerce"
        elif any(w in p_lower for w in ["student", "school", "exam", "education", "grade", "gpa", "attendance"]):
            domain = "Education"
        elif any(w in p_lower for w in ["crop", "soil", "agriculture", "farm", "yield", "rainfall"]):
            domain = "Agriculture"

        # Task extraction
        ml_task = "Classification"
        if any(w in p_lower for w in ["regress", "continuous", "predict score", "yield", "amount", "price"]):
            ml_task = "Regression"
        elif any(w in p_lower for w in ["cluster", "unsupervised", "grouping", "segmentation", "segments"]):
            ml_task = "Clustering"

        # Missing percentage
        missing_pct = 5.0
        missing_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:missing|null)", p_lower)
        if missing_match:
            try:
                missing_pct = float(missing_match.group(1))
            except ValueError:
                pass
        elif "no missing" in p_lower or "zero missing" in p_lower:
            missing_pct = 0.0

        return {
            "domain": domain,
            "ml_task": ml_task,
            "n_rows": n_rows,
            "missing_percentage": missing_pct,
            "source": "fallback_heuristic_parser",
        }
