"""Modular workflow pipeline for synthetic dataset generation.

Structured to align cleanly with LangGraph state-graph patterns for future
autonomous agent orchestration across 6 clear sequential stages:
1. Requirements & Planning
2. Schema Construction
3. Data Generation
4. Validation & Quality Audit
5. Evaluation & Repair/Retry
6. Export Preparation
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
except ImportError:
    from synthetic_dataset_generator.core.schema_builder import DatasetSchema
    from synthetic_dataset_generator.core.data_generator import DataGenerator
    from synthetic_dataset_generator.core.validator import DataValidator, QualityReport
    from synthetic_dataset_generator.core.exporter import ExportService
    from synthetic_dataset_generator.ai.schema_agent import SchemaAgent


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
    """Executes the 6-stage synthetic data generation workflow."""

    def __init__(self, schema_agent: Optional[SchemaAgent] = None):
        self.schema_agent = schema_agent or SchemaAgent()

    def run(self, initial_state: WorkflowState) -> WorkflowState:
        """Run the full end-to-end pipeline through all 6 stages."""
        state = initial_state

        try:
            state = self.stage_1_plan(state)
            state = self.stage_2_schema(state)
            state = self.stage_3_generate(state)
            state = self.stage_4_validate(state)
            state = self.stage_5_evaluate_retry(state)
            state = self.stage_6_export(state)
            state.success = True
        except Exception as exc:
            state.success = False
            state.error_message = str(exc)
            state.step_history.append(f"Failed at {state.current_step}: {str(exc)}")

        return state

    def stage_1_plan(self, state: WorkflowState) -> WorkflowState:
        """Stage 1: Requirements & Planning."""
        state.current_step = "Planning"
        state.step_history.append("Stage 1: Interpreted requirements & formulated data generation plan.")
        state.generation_plan = {
            "has_custom_prompt": state.raw_prompt is not None,
            "domain": state.schema.domain if state.schema else "Pending",
        }
        return state

    def stage_2_schema(self, state: WorkflowState) -> WorkflowState:
        """Stage 2: Schema Construction."""
        state.current_step = "Schema Building"
        if state.schema is None and state.raw_prompt:
            state.schema = self.schema_agent.create_schema_from_prompt(state.raw_prompt)
        state.step_history.append("Stage 2: Validated DatasetSchema established.")
        return state

    def stage_3_generate(self, state: WorkflowState) -> WorkflowState:
        """Stage 3: Data Generation Engine."""
        state.current_step = "Data Generation"
        if not state.schema:
            raise ValueError("Schema must be defined before generation.")
        generator = DataGenerator(state.schema)
        state.dataset = generator.generate()
        state.step_history.append(f"Stage 3: Generated {len(state.dataset)} rows across {len(state.dataset.columns)} columns.")
        return state

    def stage_4_validate(self, state: WorkflowState) -> WorkflowState:
        """Stage 4: Validation & Quality Audit."""
        state.current_step = "Quality Validation"
        if state.dataset is None or state.schema is None:
            raise ValueError("Dataset and schema must exist for validation.")
        validator = DataValidator(state.schema)
        repaired_df, report = validator.validate(state.dataset, auto_repair=True)
        state.dataset = repaired_df
        state.quality_report = report
        state.step_history.append(f"Stage 4: Quality validation complete (Score: {report.overall_score}/100).")
        return state

    def stage_5_evaluate_retry(self, state: WorkflowState) -> WorkflowState:
        """Stage 5: Evaluation & Auto-repair."""
        state.current_step = "Evaluation & Repair"
        # If any failed check occurred, report or re-validate
        if state.quality_report and state.quality_report.failed_count > 0:
            state.step_history.append("Stage 5: Unresolved validation issues flagged.")
        else:
            state.step_history.append("Stage 5: Quality criteria satisfied. Ready for export.")
        return state

    def stage_6_export(self, state: WorkflowState) -> WorkflowState:
        """Stage 6: Export Service."""
        state.current_step = "Export Preparation"
        if state.dataset is not None and state.schema is not None:
            csv_b, _, _ = ExportService.to_csv(state.dataset, state.schema.domain, state.schema.ml_task)
            state.exports["csv"] = csv_b
            state.step_history.append("Stage 6: Pre-compiled in-memory export buffers.")
        return state
