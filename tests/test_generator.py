"""Tests for Synthetic Data Generator engine."""

import numpy as np
import pandas as pd
import pytest

from synthetic_dataset_generator.core.schema_builder import (
    ColumnConfig,
    ColumnType,
    DistributionType,
    DatasetSchema,
    build_schema_for_domain,
)
from synthetic_dataset_generator.core.data_generator import DataGenerator


def test_numeric_and_categorical_generation():
    """Verify numeric features obey bounds and categorical features contain allowed categories."""
    schema = build_schema_for_domain("Healthcare", task="Classification", n_rows=250, seed=42)
    generator = DataGenerator(schema)
    df = generator.generate()

    assert len(df) == 250
    assert "Age" in df.columns
    assert "Gender" in df.columns

    # Bounds check (non-null values)
    age_clean = df["Age"].dropna()
    assert (age_clean >= 18).all()
    assert (age_clean <= 90).all()

    # Categorical allowed values
    gender_clean = df["Gender"].dropna()
    assert set(gender_clean.unique()).issubset({"Male", "Female"})


def test_classification_target_creation():
    """Verify binary classification target creation and class balance."""
    schema = build_schema_for_domain("Healthcare", task="Classification", n_rows=500, seed=123)
    schema.class_balance = {"0": 0.60, "1": 0.40}
    generator = DataGenerator(schema)
    df = generator.generate()

    assert "Disease" in df.columns
    unique_vals = set(df["Disease"].dropna().unique())
    assert unique_vals.issubset({0, 1})

    # Check class balance is approximately 40% (within 5% tolerance)
    pos_ratio = (df["Disease"] == 1).mean()
    assert 0.32 <= pos_ratio <= 0.48


def test_regression_target_creation():
    """Verify regression target continuous values and sensible ranges."""
    schema = build_schema_for_domain("Healthcare", task="Regression", n_rows=300, seed=42)
    generator = DataGenerator(schema)
    df = generator.generate()

    target = schema.target_column
    assert target is not None
    assert target in df.columns
    assert pd.api.types.is_numeric_dtype(df[target])

    # Check target range
    vals = df[target].dropna()
    assert (vals >= 1.0).all()
    assert (vals <= 60.0).all()


def test_clustering_output():
    """Verify clustering generation works without requiring a target variable."""
    schema = build_schema_for_domain("Healthcare", task="Clustering", n_rows=200, seed=42)
    generator = DataGenerator(schema)
    df = generator.generate()

    assert schema.target_column is None
    assert len(df) == 200
    assert "Patient_ID" in df.columns
    assert "BMI" in df.columns


def test_reproducibility_with_same_seed():
    """Verify bitwise identical dataset generation when using the same seed."""
    schema1 = build_schema_for_domain("Banking and Finance", task="Classification", n_rows=200, seed=777)
    schema2 = build_schema_for_domain("Banking and Finance", task="Classification", n_rows=200, seed=777)
    schema3 = build_schema_for_domain("Banking and Finance", task="Classification", n_rows=200, seed=888)

    df1 = DataGenerator(schema1).generate()
    df2 = DataGenerator(schema2).generate()
    df3 = DataGenerator(schema3).generate()

    # df1 and df2 must match identically
    pd.testing.assert_frame_equal(df1, df2)

    # df1 and df3 must not be identical
    assert not df1.equals(df3)


def test_missing_value_percentage_handling():
    """Verify missing values are injected at the configured percentage."""
    schema = build_schema_for_domain(
        "General ML",
        task="Classification",
        n_rows=1000,
        include_missing=True,
        missing_pct=10.0,
        seed=42,
    )
    generator = DataGenerator(schema)
    df = generator.generate()

    # ID column must never have nulls
    assert df["Record_ID"].isna().sum() == 0

    # Total missing cells should be approximately 10% of eligible cells
    non_id_cols = [c for c in df.columns if "ID" not in c]
    total_eligible_cells = len(df) * len(non_id_cols)
    null_cells = df[non_id_cols].isna().sum().sum()
    actual_pct = (null_cells / total_eligible_cells) * 100.0

    assert 7.0 <= actual_pct <= 13.0


def test_categorical_only_dataset():
    """Verify generator handles schema with only categorical columns without error."""
    cols = [
        ColumnConfig(name="ID", col_type=ColumnType.ID, nullable=False),
        ColumnConfig(name="Cat_1", col_type=ColumnType.CATEGORICAL, categories=["A", "B", "C"]),
        ColumnConfig(name="Cat_2", col_type=ColumnType.CATEGORICAL, categories=["X", "Y"]),
    ]
    schema = DatasetSchema(
        domain="General ML",
        ml_task="General",
        n_rows=50,
        columns=cols,
        target_column=None,
        include_missing=False,
    )
    df = DataGenerator(schema).generate()

    assert len(df) == 50
    assert set(df["Cat_1"].unique()).issubset({"A", "B", "C"})
    assert set(df["Cat_2"].unique()).issubset({"X", "Y"})
