"""Modular workflow pipeline for synthetic dataset generation.

Structured to align cleanly with LangGraph state-graph patterns
across the specialized agents:
1. Requirements & Planning
2. Knowledge Retrieval (RAG)
3. Schema Construction
4. Data Generation
5. Validation & Quality Audit
6. Evaluation & Repair/Retry Loop
7. Export Preparation
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd

try:
    from core.schema_builder import DatasetSchema
    from core.data_generator import DataGenerator
    from core.validator import DataValidator, QualityReport
    from core.exporter import ExportService
    from ai.schema_agent import SchemaAgent
    from ai.multi_agent_system import MultiAgentOrchestrator, MultiAgentState
except ImportError:
    from synthetic_dataset_generator.core.schema_builder import DatasetSchema
    from synthetic_dataset_generator.core.data_generator import DataGenerator
    from synthetic_dataset_generator.core.validator import DataValidator, QualityReport
    from synthetic_dataset_generator.core.exporter import ExportService
    from synthetic_dataset_generator.ai.schema_agent import SchemaAgent
    from synthetic_dataset_generator.ai.multi_agent_system import MultiAgentOrchestrator, MultiAgentState


@dataclass
class WorkflowState:
    """State object passed through each stage of the synthetic generation pipeline."""
    # Inputs
    raw_prompt: Optional[str] = None
    schema: Optional[DatasetSchema] = None

    # Pipeline artifacts
    generation_plan: Dict[str, Any] = field(default_factory=dict)
    dataset: Optional[pd.DataFrame] = None
    quality_report: Optional[QualityReport] = None
    exports: Dict[str, bytes] = field(default_factory=dict)

    # Status tracking
    current_step: str = "Initialized"
    step_history: List[str] = field(default_factory=list)
    success: bool = True
    error_message: Optional[str] = None


class SyntheticDataWorkflow:
    """Executes the multi-stage synthetic data generation workflow."""

    def __init__(self, schema_agent: Optional[SchemaAgent] = None):
        self.schema_agent = schema_agent or SchemaAgent()
        self.orchestrator = MultiAgentOrchestrator()

    def run(self, initial_state: WorkflowState) -> WorkflowState:
        """Run the full end-to-end pipeline through all stages."""
        # Convert WorkflowState to MultiAgentState
        agent_state = MultiAgentState(
            prompt=initial_state.raw_prompt,
            schema=initial_state.schema,
            domain=initial_state.schema.domain if initial_state.schema else "Healthcare",
            ml_task=initial_state.schema.ml_task if initial_state.schema else "Classification",
            n_rows=initial_state.schema.n_rows if initial_state.schema else 1000,
        )

        agent_state = self.orchestrator.execute(agent_state)

        initial_state.dataset = agent_state.dataset
        initial_state.schema = agent_state.schema
        initial_state.quality_report = agent_state.quality_report
        initial_state.exports = agent_state.exports
        initial_state.success = agent_state.success
        initial_state.error_message = agent_state.error_message
        initial_state.current_step = agent_state.current_state
        initial_state.step_history = [f"[{log.agent_name} | {log.state}] {log.message}" for log in agent_state.logs]

        return initial_state
