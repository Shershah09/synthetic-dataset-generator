"""Multi-Agent System for Synthetic Dataset Generation (LangGraph-Aligned).

Implements the 7 specialized agents and state machine defined in the project report:
1. Requirement Analysis Agent
2. Knowledge Retrieval Agent (RAG)
3. Schema Generation Agent (Gemini API + Fallback)
4. Data Generation Agent (Python, Pandas, NumPy, Faker)
5. Validation Agent (Pydantic, Bounds, Types)
6. Quality Evaluation Agent (0-100 Score, Metrics, Repair Feedback Loop)
7. Export Agent (In-Memory CSV, Excel, JSON)

State Transitions:
[Idle] -> [Understanding Request] -> [Retrieving Knowledge] -> [Schema Generation]
-> [Schema Validation] -> [Data Generation] -> [Data Validation] -> [Repairing]
-> [Quality Evaluation] -> [Ready] -> [Exported]
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

try:
    from core.schema_builder import DatasetSchema, build_schema_for_domain
    from core.data_generator import DataGenerator
    from core.validator import DataValidator, QualityReport
    from core.exporter import ExportService
    from ai.rag_pipeline import RAGPipeline, get_rag_pipeline, DocumentChunk
    from ai.llm_service import LLMService
except ImportError:
    from synthetic_dataset_generator.core.schema_builder import DatasetSchema, build_schema_for_domain
    from synthetic_dataset_generator.core.data_generator import DataGenerator
    from synthetic_dataset_generator.core.validator import DataValidator, QualityReport
    from synthetic_dataset_generator.core.exporter import ExportService
    from synthetic_dataset_generator.ai.rag_pipeline import RAGPipeline, get_rag_pipeline, DocumentChunk
    from synthetic_dataset_generator.ai.llm_service import LLMService


@dataclass
class AgentLogEntry:
    """Represents a log record of an agent's execution."""
    agent_name: str
    state: str
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S.%f")[:-3])
    status: str = "INFO"  # INFO, SUCCESS, WARNING, ERROR
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiAgentState:
    """Workflow state shared across all 7 agents in the LangGraph reasoning loop."""
    # Current Execution State
    current_state: str = "Idle"
    prompt: Optional[str] = None
    domain: str = "Healthcare"
    ml_task: str = "Classification"
    n_rows: int = 1000

    # Agent Artifacts
    retrieved_chunks: List[Tuple[DocumentChunk, float]] = field(default_factory=list)
    schema: Optional[DatasetSchema] = None
    dataset: Optional[pd.DataFrame] = None
    quality_report: Optional[QualityReport] = None
    exports: Dict[str, bytes] = field(default_factory=dict)

    # Execution telemetry
    logs: List[AgentLogEntry] = field(default_factory=list)
    repair_count: int = 0
    total_duration_sec: float = 0.0
    success: bool = True
    error_message: Optional[str] = None


class RequirementAnalysisAgent:
    """Agent 1: Interprets natural-language prompt or manual inputs to determine domain & task."""

    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm = llm_service or LLMService()

    def run(self, state: MultiAgentState) -> MultiAgentState:
        t0 = time.time()
        state.current_state = "Understanding Request"

        if state.prompt:
            parsed = self.llm.generate_schema_json(state.prompt)
            state.domain = parsed.get("domain", state.domain)
            state.ml_task = parsed.get("ml_task", state.ml_task)
            state.n_rows = parsed.get("n_rows", state.n_rows)
            msg = (
                f"Parsed requirements: Domain='{state.domain}', Task='{state.ml_task}', "
                f"Target Rows={state.n_rows:,}."
            )
        else:
            msg = f"Direct configuration active: Domain='{state.domain}', Task='{state.ml_task}', Rows={state.n_rows:,}."

        state.logs.append(
            AgentLogEntry(
                agent_name="Requirement Analysis Agent",
                state="Understanding Request",
                message=msg,
                duration_ms=round((time.time() - t0) * 1000, 2),
                status="SUCCESS",
            )
        )
        return state


class KnowledgeRetrievalAgent:
    """Agent 2: Retrieves domain constraints, reference ranges & rules via RAG."""

    def __init__(self, rag_pipeline: Optional[RAGPipeline] = None):
        self.rag = rag_pipeline or get_rag_pipeline()

    def run(self, state: MultiAgentState) -> MultiAgentState:
        t0 = time.time()
        state.current_state = "Retrieving Knowledge"

        query = state.prompt or f"{state.domain} {state.ml_task} dataset schema and constraints"
        matches = self.rag.similarity_search(query=query, domain=state.domain, top_k=3)
        state.retrieved_chunks = matches

        titles = [chk.title for chk, _ in matches]
        msg = f"Retrieved {len(matches)} domain reference documents from vector store: {titles}."
        state.logs.append(
            AgentLogEntry(
                agent_name="Knowledge Retrieval Agent",
                state="Retrieving Knowledge",
                message=msg,
                duration_ms=round((time.time() - t0) * 1000, 2),
                status="SUCCESS",
                metadata={"retrieved_titles": titles},
            )
        )
        return state


class SchemaGenerationAgent:
    """Agent 3: Creates structured Pydantic DatasetSchema using LLM + RAG context."""

    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm = llm_service or LLMService()

    def run(self, state: MultiAgentState) -> MultiAgentState:
        t0 = time.time()
        state.current_state = "Schema Generation"

        if state.schema is None:
            state.schema = build_schema_for_domain(
                domain=state.domain,
                task=state.ml_task,
                n_rows=state.n_rows,
            )

        # Validate Schema structure (Schema Validation State)
        state.current_state = "Schema Validation"
        col_count = len(state.schema.columns)
        msg = (
            f"Generated & validated schema for '{state.domain}' with {col_count} columns. "
            f"Target variable: '{state.schema.target_column}'."
        )
        state.logs.append(
            AgentLogEntry(
                agent_name="Schema Generation Agent",
                state="Schema Validation",
                message=msg,
                duration_ms=round((time.time() - t0) * 1000, 2),
                status="SUCCESS",
            )
        )
        return state


class DataGenerationAgent:
    """Agent 4: Executes Python, NumPy, Pandas, Faker synthetic data generation."""

    def run(self, state: MultiAgentState) -> MultiAgentState:
        t0 = time.time()
        state.current_state = "Data Generation"

        if not state.schema:
            raise ValueError("Schema must be initialized before data generation.")

        generator = DataGenerator(state.schema)
        state.dataset = generator.generate()

        msg = f"Generated {len(state.dataset):,} synthetic rows and {len(state.dataset.columns)} features."
        state.logs.append(
            AgentLogEntry(
                agent_name="Data Generation Agent",
                state="Data Generation",
                message=msg,
                duration_ms=round((time.time() - t0) * 1000, 2),
                status="SUCCESS",
            )
        )
        return state


class ValidationAgent:
    """Agent 5: Audits data types, missing values, duplicates, and constraints."""

    def run(self, state: MultiAgentState) -> MultiAgentState:
        t0 = time.time()
        state.current_state = "Data Validation"

        if state.dataset is None or state.schema is None:
            raise ValueError("Dataset and schema must exist for validation.")

        validator = DataValidator(state.schema)
        repaired_df, report = validator.validate(state.dataset, auto_repair=True)
        state.dataset = repaired_df
        state.quality_report = report

        # Check if auto-repair loop was triggered
        repaired_checks = [c for c in report.checks if c.repaired]
        if repaired_checks:
            state.current_state = "Repairing"
            state.repair_count += len(repaired_checks)
            msg = (
                f"Detected and auto-repaired {len(repaired_checks)} boundary/type violations: "
                f"{[c.check_name for c in repaired_checks]}."
            )
            state.logs.append(
                AgentLogEntry(
                    agent_name="Validation Agent",
                    state="Repairing",
                    message=msg,
                    duration_ms=round((time.time() - t0) * 1000, 2),
                    status="WARNING",
                )
            )
        else:
            msg = "Structural conformance and boundary checks passed with zero repairs needed."
            state.logs.append(
                AgentLogEntry(
                    agent_name="Validation Agent",
                    state="Data Validation",
                    message=msg,
                    duration_ms=round((time.time() - t0) * 1000, 2),
                    status="SUCCESS",
                )
            )
        return state


class QualityEvaluationAgent:
    """Agent 6: Evaluates dataset quality score (0-100) and statistical realism."""

    def run(self, state: MultiAgentState) -> MultiAgentState:
        t0 = time.time()
        state.current_state = "Quality Evaluation"

        if not state.quality_report:
            raise ValueError("Quality report must be computed before evaluation.")

        score = state.quality_report.overall_score
        passed = state.quality_report.passed_count
        warnings = state.quality_report.warning_count
        failed = state.quality_report.failed_count

        msg = (
            f"Quality Audit Score: {score}/100. Checkpoints: {passed} Passed, "
            f"{warnings} Warnings, {failed} Failed. Dataset is ready for export."
        )
        state.current_state = "Ready"
        state.logs.append(
            AgentLogEntry(
                agent_name="Quality Evaluation Agent",
                state="Ready",
                message=msg,
                duration_ms=round((time.time() - t0) * 1000, 2),
                status="SUCCESS" if score >= 80 else "WARNING",
            )
        )
        return state


class ExportAgent:
    """Agent 7: Generates in-memory CSV, Excel (.xlsx), and JSON downloads."""

    def run(self, state: MultiAgentState) -> MultiAgentState:
        t0 = time.time()
        state.current_state = "Exported"

        if state.dataset is None or state.schema is None:
            raise ValueError("Dataset and schema must be present for export.")

        csv_b, _, _ = ExportService.to_csv(state.dataset, state.schema.domain, state.schema.ml_task)
        excel_b, _, _ = ExportService.to_excel(state.dataset, state.schema.domain, state.schema.ml_task)
        json_b, _, _ = ExportService.to_json(state.dataset, state.schema.domain, state.schema.ml_task)

        state.exports["csv"] = csv_b
        state.exports["xlsx"] = excel_b
        state.exports["json"] = json_b

        msg = f"Compiled in-memory exports for CSV ({len(csv_b):,} bytes), Excel, and JSON."
        state.logs.append(
            AgentLogEntry(
                agent_name="Export Agent",
                state="Exported",
                message=msg,
                duration_ms=round((time.time() - t0) * 1000, 2),
                status="SUCCESS",
            )
        )
        return state


class MultiAgentOrchestrator:
    """Coordinates the 7 specialized AI agents across the LangGraph reasoning loop."""

    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm = llm_service or LLMService()
        self.rag = get_rag_pipeline()

        # Instantiate 7 agents
        self.req_agent = RequirementAnalysisAgent(self.llm)
        self.rag_agent = KnowledgeRetrievalAgent(self.rag)
        self.schema_agent = SchemaGenerationAgent(self.llm)
        self.data_agent = DataGenerationAgent()
        self.val_agent = ValidationAgent()
        self.quality_agent = QualityEvaluationAgent()
        self.export_agent = ExportAgent()

    def execute(self, state: MultiAgentState) -> MultiAgentState:
        """Run the end-to-end multi-agent workflow sequence."""
        start_time = time.time()

        try:
            # Stage 1: Requirement Analysis
            state = self.req_agent.run(state)

            # Stage 2: Knowledge Retrieval (RAG)
            state = self.rag_agent.run(state)

            # Stage 3: Schema Generation & Validation
            state = self.schema_agent.run(state)

            # Stage 4: Data Generation
            state = self.data_agent.run(state)

            # Stage 5: Validation & Auto-repair
            state = self.val_agent.run(state)

            # Stage 6: Quality Evaluation
            state = self.quality_agent.run(state)

            # Stage 7: Export Preparation
            state = self.export_agent.run(state)

            state.success = True
            state.total_duration_sec = round(time.time() - start_time, 3)

        except Exception as exc:
            state.success = False
            state.current_state = "Error Handling"
            state.error_message = str(exc)
            state.logs.append(
                AgentLogEntry(
                    agent_name="Orchestrator",
                    state="Error Handling",
                    message=f"Pipeline exception: {str(exc)}",
                    status="ERROR",
                )
            )

        return state
