"""Tests for Data Validator, Quality Evaluation, and Exporter engines."""

import io
import json
import pandas as pd
import pytest
from pydantic import ValidationError

try:
    from core.schema_builder import (
        ColumnConfig,
        ColumnType,
        DatasetSchema,
        build_schema_for_domain,
    )
    from core.data_generator import DataGenerator
    from core.validator import (
        DataValidator,
        CheckStatus,
    )
    from core.exporter import ExportService
except ImportError:
    from synthetic_dataset_generator.core.schema_builder import (
        ColumnConfig,
        ColumnType,
        DatasetSchema,
        build_schema_for_domain,
    )
    from synthetic_dataset_generator.core.data_generator import DataGenerator
    from synthetic_dataset_generator.core.validator import (
        DataValidator,
        CheckStatus,
    )
    from synthetic_dataset_generator.core.exporter import ExportService


def test_invalid_schema_duplicate_columns():
    """Verify schema raises error on duplicate column names."""
    with pytest.raises(ValidationError):
        DatasetSchema(
            domain="Test",
            ml_task="Classification",
            n_rows=100,
            columns=[
                ColumnConfig(name="Feature_A", col_type=ColumnType.NUMERIC),
                ColumnConfig(name="Feature_A", col_type=ColumnType.NUMERIC),
            ],
        )


def test_validator_detects_and_repairs_bounds():
    """Verify validator flags out-of-bound values and auto-repairs them via clipping."""
    schema = build_schema_for_domain("Healthcare", n_rows=100, seed=42)
    generator = DataGenerator(schema)
    df = generator.generate()

    # Manually inject an out-of-bounds value into Age (min=18, max=90)
    df.loc[0, "Age"] = 999.0

    validator = DataValidator(schema)
    repaired_df, report = validator.validate(df, auto_repair=True)

    # After auto-repair, max Age must be <= 90
    assert repaired_df["Age"].max() <= 90.0

    # Report should indicate check repaired or warning
    bound_checks = [c for c in report.checks if c.check_name == "Numeric Range Constraints"]
    assert len(bound_checks) == 1
    assert bound_checks[0].repaired is True


def test_validator_quality_report():
    """Verify quality score computation and check reporting."""
    schema = build_schema_for_domain("Banking and Finance", n_rows=150, seed=42)
    df = DataGenerator(schema).generate()

    validator = DataValidator(schema)
    _, report = validator.validate(df, auto_repair=True)

    assert 0 <= report.overall_score <= 100
    assert report.is_download_ready is True
    assert report.passed_count > 0
    assert len(report.checks) >= 5


def test_exporter_csv_excel_json():
    """Verify CSV, Excel, and JSON exports produce valid non-empty files preserving all rows."""
    schema = build_schema_for_domain("Education", n_rows=100, seed=42)
    df = DataGenerator(schema).generate()

    # 1. Test CSV export
    csv_bytes, csv_mime, csv_name = ExportService.to_csv(df, "Education", "Classification")
    assert len(csv_bytes) > 0
    assert csv_mime == "text/csv"
    assert csv_name.endswith(".csv")

    reloaded_csv = pd.read_csv(io.BytesIO(csv_bytes))
    assert len(reloaded_csv) == 100
    assert set(reloaded_csv.columns) == set(df.columns)

    # 2. Test Excel export
    excel_bytes, excel_mime, excel_name = ExportService.to_excel(df, "Education", "Classification")
    assert len(excel_bytes) > 0
    assert "openxmlformats" in excel_mime
    assert excel_name.endswith(".xlsx")

    reloaded_excel = pd.read_excel(io.BytesIO(excel_bytes))
    assert len(reloaded_excel) == 100
    assert set(reloaded_excel.columns) == set(df.columns)

    # 3. Test JSON export
    json_bytes, json_mime, json_name = ExportService.to_json(df, "Education", "Classification")
    assert len(json_bytes) > 0
    assert json_mime == "application/json"
    assert json_name.endswith(".json")

    parsed_json = json.loads(json_bytes.decode("utf-8"))
    assert len(parsed_json) == 100
    assert isinstance(parsed_json, list)
