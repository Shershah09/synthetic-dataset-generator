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
from typing import Any, Dict, List, Optional, Tuple, TypedDict
import pandas as pd

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = "__end__"

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


class AgentGraphState(TypedDict, total=False):
    """LangGraph State schema dictionary for graph execution."""
    current_state: str
    prompt: Optional[str]
    domain: str
    ml_task: str
    n_rows: int
    retrieved_chunks: List[Any]
    schema: Optional[Any]
    dataset: Optional[Any]
    quality_report: Optional[Any]
    exports: Dict[str, bytes]
    logs: List[AgentLogEntry]
    repair_count: int
    repair_needed: bool
    total_duration_sec: float
    success: bool
    error_message: Optional[str]
    is_langgraph_native: bool


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
    is_langgraph_native: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert state into a LangGraph-compatible dictionary."""
        return {
            "current_state": self.current_state,
            "prompt": self.prompt,
            "domain": self.domain,
            "ml_task": self.ml_task,
            "n_rows": self.n_rows,
            "retrieved_chunks": list(self.retrieved_chunks),
            "schema": self.schema,
            "dataset": self.dataset,
            "quality_report": self.quality_report,
            "exports": dict(self.exports),
            "logs": list(self.logs),
            "repair_count": self.repair_count,
            "repair_needed": False,
            "total_duration_sec": self.total_duration_sec,
            "success": self.success,
            "error_message": self.error_message,
            "is_langgraph_native": self.is_langgraph_native,
        }

    def update_from_dict(self, data: Dict[str, Any]) -> MultiAgentState:
        """Update state fields in-place from LangGraph execution output."""
        for k, v in data.items():
            if hasattr(self, k):
                setattr(self, k, v)
        return self


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
    """Agent 3: Creates structured Pydantic DatasetSchema using LangChain + RAG context."""

    def __init__(self, llm_service: Optional[LLMService] = None, rag_pipeline: Optional[RAGPipeline] = None):
        self.llm = llm_service or LLMService()
        self.rag = rag_pipeline or get_rag_pipeline()

    def run(self, state: MultiAgentState) -> MultiAgentState:
        t0 = time.time()
        state.current_state = "Schema Generation"

        if state.schema is None:
            if state.prompt:
                rag_context = self.rag.build_augmented_context(state.prompt, domain=state.domain)
                # Invoking LangChain PromptTemplate + LCEL Chain
                parsed = self.llm.invoke_langchain(state.prompt, rag_context=rag_context)
                state.domain = parsed.get("domain", state.domain)
                state.ml_task = parsed.get("ml_task", state.ml_task)
                state.n_rows = parsed.get("n_rows", state.n_rows)

            state.schema = build_schema_for_domain(
                domain=state.domain,
                task=state.ml_task,
                n_rows=state.n_rows,
            )

        # Validate Schema structure (Schema Validation State)
        state.current_state = "Schema Validation"
        col_count = len(state.schema.columns)
        msg = (
            f"Generated & validated schema via LangChain for '{state.domain}' with {col_count} columns. "
            f"Target variable: '{state.schema.target_column}'."
        )
        state.logs.append(
            AgentLogEntry(
                agent_name="Schema Generation Agent (LangChain)",
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
    """Coordinates the 7 specialized AI agents across the LangGraph StateGraph reasoning loop."""

    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm = llm_service or LLMService()
        self.rag = get_rag_pipeline()

        # Instantiate 7 specialized agents
        self.req_agent = RequirementAnalysisAgent(self.llm)
        self.rag_agent = KnowledgeRetrievalAgent(self.rag)
        self.schema_agent = SchemaGenerationAgent(self.llm, self.rag)
        self.data_agent = DataGenerationAgent()
        self.val_agent = ValidationAgent()
        self.quality_agent = QualityEvaluationAgent()
        self.export_agent = ExportAgent()

        # Compile LangGraph StateGraph if available
        self.langgraph_app = self._build_langgraph()

    def _build_langgraph(self) -> Any:
        """Constructs and compiles the formal LangGraph StateGraph with self-healing conditional loop."""
        if not LANGGRAPH_AVAILABLE:
            return None

        workflow = StateGraph(AgentGraphState)

        # 1. Define Graph Node Wrappers
        def node_requirement_analysis(state_dict: AgentGraphState) -> Dict[str, Any]:
            s = MultiAgentState().update_from_dict(state_dict)
            s = self.req_agent.run(s)
            return s.to_dict()

        def node_knowledge_retrieval(state_dict: AgentGraphState) -> Dict[str, Any]:
            s = MultiAgentState().update_from_dict(state_dict)
            s = self.rag_agent.run(s)
            return s.to_dict()

        def node_schema_generation(state_dict: AgentGraphState) -> Dict[str, Any]:
            s = MultiAgentState().update_from_dict(state_dict)
            s = self.schema_agent.run(s)
            return s.to_dict()

        def node_data_generation(state_dict: AgentGraphState) -> Dict[str, Any]:
            s = MultiAgentState().update_from_dict(state_dict)
            s = self.data_agent.run(s)
            return s.to_dict()

        def node_validation(state_dict: AgentGraphState) -> Dict[str, Any]:
            s = MultiAgentState().update_from_dict(state_dict)
            prev_repairs = state_dict.get("repair_count", 0)
            s = self.val_agent.run(s)
            out = s.to_dict()
            # If new repairs occurred and we haven't hit loop limit, route to repair loop
            if s.repair_count > prev_repairs and s.repair_count <= 2:
                out["repair_needed"] = True
            else:
                out["repair_needed"] = False
            return out

        def node_repair(state_dict: AgentGraphState) -> Dict[str, Any]:
            s = MultiAgentState().update_from_dict(state_dict)
            if s.dataset is not None and s.schema is not None:
                validator = DataValidator(s.schema)
                repaired_df, report = validator.validate(s.dataset, auto_repair=True)
                s.dataset = repaired_df
                s.quality_report = report
                s.logs.append(
                    AgentLogEntry(
                        agent_name="Validation Agent (LangGraph Self-Healing Loop)",
                        state="Repairing",
                        message=f"LangGraph cyclical repair executed. Refined Quality Score: {report.overall_score}/100.",
                        status="SUCCESS",
                    )
                )
            out = s.to_dict()
            out["repair_needed"] = False
            return out

        def node_quality_evaluation(state_dict: AgentGraphState) -> Dict[str, Any]:
            s = MultiAgentState().update_from_dict(state_dict)
            s = self.quality_agent.run(s)
            return s.to_dict()

        def node_export(state_dict: AgentGraphState) -> Dict[str, Any]:
            s = MultiAgentState().update_from_dict(state_dict)
            s = self.export_agent.run(s)
            return s.to_dict()

        # 2. Add All Nodes to StateGraph
        workflow.add_node("requirement_analysis", node_requirement_analysis)
        workflow.add_node("knowledge_retrieval", node_knowledge_retrieval)
        workflow.add_node("schema_generation", node_schema_generation)
        workflow.add_node("data_generation", node_data_generation)
        workflow.add_node("validation", node_validation)
        workflow.add_node("repair", node_repair)
        workflow.add_node("quality_evaluation", node_quality_evaluation)
        workflow.add_node("export", node_export)

        # 3. Add Edges and Conditional Router
        workflow.set_entry_point("requirement_analysis")
        workflow.add_edge("requirement_analysis", "knowledge_retrieval")
        workflow.add_edge("knowledge_retrieval", "schema_generation")
        workflow.add_edge("schema_generation", "data_generation")
        workflow.add_edge("data_generation", "validation")

        def route_validation_outcome(state_dict: AgentGraphState) -> str:
            """Conditional edge: Route to self-repair node if violations detected."""
            if state_dict.get("repair_needed", False) and state_dict.get("repair_count", 0) <= 2:
                return "repair"
            return "quality_evaluation"

        workflow.add_conditional_edges(
            "validation",
            route_validation_outcome,
            {
                "repair": "repair",
                "quality_evaluation": "quality_evaluation",
            },
        )
        workflow.add_edge("repair", "quality_evaluation")
        workflow.add_edge("quality_evaluation", "export")
        workflow.add_edge("export", END)

        return workflow.compile()

    def get_mermaid_graph(self) -> str:
        """Returns Mermaid diagram string of the compiled LangGraph workflow."""
        if self.langgraph_app is not None:
            try:
                return self.langgraph_app.get_graph().draw_mermaid()
            except Exception:
                pass
        return """graph TD
    __start__([Start]) --> requirement_analysis[1. Requirement Analysis Agent]
    requirement_analysis --> knowledge_retrieval[2. Knowledge Retrieval Agent - RAG]
    knowledge_retrieval --> schema_generation[3. Schema Generation Agent - LangChain]
    schema_generation --> data_generation[4. Data Generation Agent]
    data_generation --> validation{5. Validation Agent}
    validation -->|Repair Needed| repair[Self-Healing Repair Loop]
    repair --> quality_evaluation[6. Quality Evaluation Agent]
    validation -->|Passed| quality_evaluation
    quality_evaluation --> export[7. Export Agent]
    export --> __end__([Exported Datasets])
"""

    def execute(self, state: MultiAgentState) -> MultiAgentState:
        """Run the end-to-end multi-agent workflow sequence via LangGraph."""
        start_time = time.time()

        # Branch 1: Native LangGraph execution if compiled
        if self.langgraph_app is not None:
            try:
                state_dict = state.to_dict()
                result_dict = self.langgraph_app.invoke(state_dict)
                state.update_from_dict(result_dict)
                state.is_langgraph_native = True
                state.total_duration_sec = round(time.time() - start_time, 3)
                state.success = True
                return state
            except Exception as exc:
                # Log error and fall back gracefully to sequential pipeline
                state.logs.append(
                    AgentLogEntry(
                        agent_name="LangGraph Orchestrator",
                        state="Fallback",
                        message=f"LangGraph execution exception ({str(exc)}); invoking sequential fallback.",
                        status="WARNING",
                    )
                )

        # Branch 2: Robust Fallback Sequential Runner (Offline or LangGraph Unavailable)
        try:
            state = self.req_agent.run(state)
            state = self.rag_agent.run(state)
            state = self.schema_agent.run(state)
            state = self.data_agent.run(state)
            state = self.val_agent.run(state)
            state = self.quality_agent.run(state)
            state = self.export_agent.run(state)

            state.success = True
            state.is_langgraph_native = False
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
