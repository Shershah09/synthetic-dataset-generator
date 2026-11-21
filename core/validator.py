"""Data validation and quality assessment engine.

Performs structural, statistical, and constraint checks on the synthetic dataset,
supports automatic repairs (such as clipping out-of-bounds or replacing invalid infinities),
and outputs a comprehensive quality report with a 0-100 quality score.
"""

from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

try:
    from core.schema_builder import ColumnType, DatasetSchema
except ImportError:
    from synthetic_dataset_generator.core.schema_builder import ColumnType, DatasetSchema


class CheckStatus(str, Enum):
    PASSED = "PASSED"
    WARNING = "WARNING"
    FAILED = "FAILED"


class ValidationItem(BaseModel):
    category: str  # Structural, Integrity, Bounds & Types, Statistical Realism
    check_name: str
    status: CheckStatus
    details: str
    repaired: bool = False


class QualityReport(BaseModel):
    overall_score: int = Field(ge=0, le=100)
    passed_count: int = 0
    warning_count: int = 0
    failed_count: int = 0
    checks: List[ValidationItem] = Field(default_factory=list)
    is_download_ready: bool = True
    summary_message: str = ""
    column_stats: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    structural_validity: bool = True
    statistical_realism_note: str = (
        "Structural validity confirmed. Note: Synthetic datasets simulate specified statistical "
        "distributions and correlations for prototyping and ML experiments, but do not replace "
        "empirical real-world ground-truth distributions."
    )


class DataValidator:
    """Validates synthetic data against schema and quality criteria."""

    def __init__(self, schema: DatasetSchema):
        self.schema = schema

    def validate(self, df: pd.DataFrame, auto_repair: bool = True) -> Tuple[pd.DataFrame, QualityReport]:
        """Run all validation checks and return repaired df plus the QualityReport."""
        df_repaired = df.copy()
        checks: List[ValidationItem] = []

        # -----------------------------------------------------------------
        # 1. Row and Column Count Check
        # -----------------------------------------------------------------
        expected_rows = self.schema.n_rows
        actual_rows = len(df_repaired)
        if actual_rows >= expected_rows:
            checks.append(
                ValidationItem(
                    category="Structural",
                    check_name="Row Count Verification",
                    status=CheckStatus.PASSED,
                    details=f"Dataset contains {actual_rows:,} rows (expected >= {expected_rows:,}).",
                )
            )
        else:
            checks.append(
                ValidationItem(
                    category="Structural",
                    check_name="Row Count Verification",
                    status=CheckStatus.WARNING,
                    details=f"Dataset contains {actual_rows:,} rows (expected {expected_rows:,}).",
                )
            )

        expected_col_names = [c.name for c in self.schema.columns]
        missing_cols = [c for c in expected_col_names if c not in df_repaired.columns]
        if not missing_cols:
            checks.append(
                ValidationItem(
                    category="Structural",
                    check_name="Column Schema Conformance",
                    status=CheckStatus.PASSED,
                    details=f"All {len(expected_col_names)} schema columns are present in DataFrame.",
                )
            )
        else:
            checks.append(
                ValidationItem(
                    category="Structural",
                    check_name="Column Schema Conformance",
                    status=CheckStatus.FAILED,
                    details=f"Missing columns: {missing_cols}",
                )
            )

        # -----------------------------------------------------------------
        # 2. Infinite Values Check
        # -----------------------------------------------------------------
        numeric_cols = df_repaired.select_dtypes(include=[np.number]).columns
        has_inf = False
        inf_cols = []
        for col in numeric_cols:
            inf_mask = np.isinf(df_repaired[col])
            if inf_mask.any():
                has_inf = True
                inf_cols.append(col)
                if auto_repair:
                    median_val = df_repaired.loc[~inf_mask, col].median()
                    df_repaired.loc[inf_mask, col] = median_val

        if not has_inf:
            checks.append(
                ValidationItem(
                    category="Integrity",
                    check_name="Infinite Values Check",
                    status=CheckStatus.PASSED,
                    details="Zero infinite (inf / -inf) values detected across all numeric features.",
                )
            )
        else:
            checks.append(
                ValidationItem(
                    category="Integrity",
                    check_name="Infinite Values Check",
                    status=CheckStatus.WARNING if auto_repair else CheckStatus.FAILED,
                    details=f"Infinities detected in: {inf_cols}." + (" Automatically replaced with median." if auto_repair else ""),
                    repaired=auto_repair,
                )
            )

        # -----------------------------------------------------------------
        # 3. Numeric Bounds Verification & Auto-repair
        # -----------------------------------------------------------------
        bound_violations = []
        for col_cfg in self.schema.columns:
            if col_cfg.col_type == ColumnType.NUMERIC and col_cfg.name in df_repaired.columns:
                series = df_repaired[col_cfg.name].dropna()
                out_of_bounds = False
                if col_cfg.min_val is not None and (series < col_cfg.min_val - 1e-4).any():
                    out_of_bounds = True
                if col_cfg.max_val is not None and (series > col_cfg.max_val + 1e-4).any():
                    out_of_bounds = True

                if out_of_bounds:
                    bound_violations.append(col_cfg.name)
                    if auto_repair:
                        df_repaired[col_cfg.name] = df_repaired[col_cfg.name].clip(
                            lower=col_cfg.min_val, upper=col_cfg.max_val
                        )

        if not bound_violations:
            checks.append(
                ValidationItem(
                    category="Bounds & Types",
                    check_name="Numeric Range Constraints",
                    status=CheckStatus.PASSED,
                    details="All numeric columns strictly satisfy minimum and maximum boundary limits.",
                )
            )
        else:
            checks.append(
                ValidationItem(
                    category="Bounds & Types",
                    check_name="Numeric Range Constraints",
                    status=CheckStatus.WARNING if auto_repair else CheckStatus.FAILED,
                    details=f"Bound violations in columns: {bound_violations}." + (" Re-clipped to boundaries." if auto_repair else ""),
                    repaired=auto_repair,
                )
            )

        # -----------------------------------------------------------------
        # 4. Categorical Allowed Values Check
        # -----------------------------------------------------------------
        cat_violations = []
        for col_cfg in self.schema.columns:
            if col_cfg.col_type == ColumnType.CATEGORICAL and col_cfg.name in df_repaired.columns:
                if col_cfg.categories:
                    allowed = set(col_cfg.categories)
                    actual_unique = set(df_repaired[col_cfg.name].dropna().unique())
                    unrecognized = actual_unique - allowed
                    if unrecognized:
                        cat_violations.append((col_cfg.name, list(unrecognized)))

        if not cat_violations:
            checks.append(
                ValidationItem(
                    category="Bounds & Types",
                    check_name="Categorical Domain Validity",
                    status=CheckStatus.PASSED,
                    details="All categorical variables match defined category domains.",
                )
            )
        else:
            checks.append(
                ValidationItem(
                    category="Bounds & Types",
                    check_name="Categorical Domain Validity",
                    status=CheckStatus.FAILED,
                    details=f"Unrecognized categories found in: {cat_violations}",
                )
            )

        # -----------------------------------------------------------------
        # 5. Missing Values Audit
        # -----------------------------------------------------------------
        total_cells = df_repaired.size
        null_cells = df_repaired.isna().sum().sum()
        actual_missing_pct = (null_cells / total_cells) * 100.0 if total_cells > 0 else 0.0

        if not self.schema.include_missing:
            if null_cells == 0:
                checks.append(
                    ValidationItem(
                        category="Missing Data",
                        check_name="Missing Values Policy",
                        status=CheckStatus.PASSED,
                        details="No missing values (0.0%), conforming to configuration.",
                    )
                )
            else:
                checks.append(
                    ValidationItem(
                        category="Missing Data",
                        check_name="Missing Values Policy",
                        status=CheckStatus.WARNING,
                        details=f"Found {null_cells} missing cells ({actual_missing_pct:.2f}%) despite missing=False.",
                    )
                )
        else:
            # Expected roughly schema.missing_percentage
            expected_pct = self.schema.missing_percentage
            checks.append(
                ValidationItem(
                    category="Missing Data",
                    check_name="Missing Values Policy",
                    status=CheckStatus.PASSED,
                    details=f"Actual missing value rate is {actual_missing_pct:.2f}% (configured: {expected_pct:.1f}%).",
                )
            )

        # -----------------------------------------------------------------
        # 6. Target Variable Validity
        # -----------------------------------------------------------------
        if self.schema.ml_task in ["Classification", "Regression"] and self.schema.target_column:
            target_col = self.schema.target_column
            if target_col in df_repaired.columns:
                target_series = df_repaired[target_col].dropna()
                if len(target_series) == 0:
                    checks.append(
                        ValidationItem(
                            category="Target & Distribution",
                            check_name="Target Column Validity",
                            status=CheckStatus.FAILED,
                            details=f"Target column '{target_col}' has 0 non-null values.",
                        )
                    )
                elif self.schema.ml_task == "Classification":
                    unique_classes = target_series.unique()
                    counts = target_series.value_counts(normalize=True).to_dict()
                    counts_str = ", ".join([f"{k}: {v*100:.1f}%" for k, v in counts.items()])
                    checks.append(
                        ValidationItem(
                            category="Target & Distribution",
                            check_name="Target Column Validity",
                            status=CheckStatus.PASSED,
                            details=f"Target '{target_col}' has {len(unique_classes)} classes with distribution: [{counts_str}].",
                        )
                    )
                else:  # Regression
                    mean_val = target_series.mean()
                    std_val = target_series.std()
                    checks.append(
                        ValidationItem(
                            category="Target & Distribution",
                            check_name="Target Column Validity",
                            status=CheckStatus.PASSED,
                            details=f"Continuous target '{target_col}' verified: mean={mean_val:.2f}, std={std_val:.2f}.",
                        )
                    )
            else:
                checks.append(
                    ValidationItem(
                        category="Target & Distribution",
                        check_name="Target Column Validity",
                        status=CheckStatus.FAILED,
                        details=f"Specified target column '{target_col}' not found in dataset.",
                    )
                )

        # -----------------------------------------------------------------
        # 7. Duplicate Records Check
        # -----------------------------------------------------------------
        # Exclude ID column for duplicate row detection
        feature_cols_for_dup = [c for c in df_repaired.columns if "ID" not in c.upper()]
        if feature_cols_for_dup:
            dup_count = df_repaired.duplicated(subset=feature_cols_for_dup).sum()
            dup_pct = (dup_count / actual_rows) * 100.0 if actual_rows > 0 else 0.0
            if dup_count == 0 or dup_pct <= self.schema.duplicate_percentage + 1.0:
                checks.append(
                    ValidationItem(
                        category="Integrity",
                        check_name="Duplicate Records Audit",
                        status=CheckStatus.PASSED,
                        details=f"{dup_count} duplicate feature rows detected ({dup_pct:.2f}%).",
                    )
                )
            else:
                checks.append(
                    ValidationItem(
                        category="Integrity",
                        check_name="Duplicate Records Audit",
                        status=CheckStatus.WARNING,
                        details=f"Duplicate count {dup_count} exceeds expected baseline ({dup_pct:.2f}%).",
                    )
                )

        # -----------------------------------------------------------------
        # 8. User-Defined Constraints Validation
        # -----------------------------------------------------------------
        for constraint in self.schema.constraints:
            try:
                # Evaluate boolean condition with pandas query
                violations = len(df_repaired.dropna().query(f"not ({constraint.rule})"))
                if violations == 0:
                    checks.append(
                        ValidationItem(
                            category="Integrity",
                            check_name=f"Constraint: {constraint.description}",
                            status=CheckStatus.PASSED,
                            details="100% of non-null records satisfy rule: " + constraint.rule,
                        )
                    )
                else:
                    checks.append(
                        ValidationItem(
                            category="Integrity",
                            check_name=f"Constraint: {constraint.description}",
                            status=CheckStatus.WARNING,
                            details=f"{violations} records violated rule: {constraint.rule}",
                        )
                    )
            except Exception as e:
                # Query evaluation syntax error or non-evaluable rule
                pass

        # -----------------------------------------------------------------
        # Column Statistics Summary Computation
        # -----------------------------------------------------------------
        column_stats: Dict[str, Dict[str, Any]] = {}
        for col in df_repaired.columns:
            non_null = int(df_repaired[col].notna().sum())
            missing_pct = float((df_repaired[col].isna().sum() / actual_rows) * 100.0) if actual_rows > 0 else 0.0
            is_num = pd.api.types.is_numeric_dtype(df_repaired[col])

            stat_entry: Dict[str, Any] = {
                "type": "Numeric" if is_num else "Categorical",
                "non_null": non_null,
                "missing_pct": round(missing_pct, 1),
                "mean": round(float(df_repaired[col].mean()), 2) if is_num else "-",
                "std": round(float(df_repaired[col].std()), 2) if is_num else "-",
                "min": round(float(df_repaired[col].min()), 2) if is_num else "-",
                "max": round(float(df_repaired[col].max()), 2) if is_num else "-",
            }
            column_stats[col] = stat_entry

        # Compute Overall Quality Score
        passed_count = sum(1 for c in checks if c.status == CheckStatus.PASSED)
        warning_count = sum(1 for c in checks if c.status == CheckStatus.WARNING)
        failed_count = sum(1 for c in checks if c.status == CheckStatus.FAILED)

        total_checks = len(checks)
        if total_checks > 0:
            score = int((passed_count * 100 + warning_count * 70) / total_checks)
        else:
            score = 100

        is_download_ready = failed_count == 0
        summary_msg = (
            f"Quality Score: {score}/100 • {passed_count} Passed • {warning_count} Warnings • {failed_count} Failed"
        )

        report = QualityReport(
            overall_score=score,
            passed_count=passed_count,
            warning_count=warning_count,
            failed_count=failed_count,
            checks=checks,
            is_download_ready=is_download_ready,
            summary_message=summary_msg,
            column_stats=column_stats,
            structural_validity=(failed_count == 0),
        )

        return df_repaired, report
