"""Schema Agent: Converts requirements into verified DatasetSchema instances.

Works in conjunction with LLMService to produce robust, validated schemas.
"""

from __future__ import annotations
from typing import Optional
from synthetic_dataset_generator.core.schema_builder import (
    DatasetSchema,
    build_schema_for_domain,
)
from synthetic_dataset_generator.ai.llm_service import LLMService


class SchemaAgent:
    """Agent that interprets natural language user requests and returns a valid DatasetSchema."""

    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm = llm_service or LLMService()

    def create_schema_from_prompt(self, user_prompt: str) -> DatasetSchema:
        """Parse natural language request and build a corresponding DatasetSchema."""
        parsed = self.llm.generate_schema_json(user_prompt)

        domain = parsed.get("domain", "Healthcare")
        ml_task = parsed.get("ml_task", "Classification")
        n_rows = parsed.get("n_rows", 1000)
        missing_pct = parsed.get("missing_percentage", 5.0)

        schema = build_schema_for_domain(
            domain=domain,
            task=ml_task,
            n_rows=n_rows,
            include_missing=(missing_pct > 0),
            missing_pct=missing_pct,
        )

        return schema
