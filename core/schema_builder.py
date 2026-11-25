"""Schema builder and data models for synthetic dataset generation.

Defines Pydantic models for column types, distributions, constraints, and
domain presets (Healthcare, Banking, E-commerce, Education, Agriculture, General ML).
"""

from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class ColumnType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    ID = "id"


class DistributionType(str, Enum):
    NORMAL = "normal"
    UNIFORM = "uniform"
    EXPONENTIAL = "exponential"
    LOGNORMAL = "lognormal"
    POISSON = "poisson"
    BIMODAL = "bimodal"


class ColumnConfig(BaseModel):
    name: str
    col_type: ColumnType = ColumnType.NUMERIC
    distribution: Optional[DistributionType] = DistributionType.NORMAL
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    categories: Optional[List[str]] = None
    category_weights: Optional[List[float]] = None
    is_target: bool = False
    faker_provider: Optional[str] = None  # e.g., 'name', 'email', 'date', etc.
    date_start: Optional[str] = "2023-01-01"
    date_end: Optional[str] = "2024-12-31"
    decimals: int = 2
    nullable: bool = True

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("Column name cannot be empty")
        return clean


class ConstraintConfig(BaseModel):
    rule: str
    description: str


class DatasetSchema(BaseModel):
    domain: str = "Healthcare"
    ml_task: str = "Classification"  # Classification, Regression, Clustering, General
    n_rows: int = Field(default=1000, ge=1, le=100000)
    columns: List[ColumnConfig] = Field(default_factory=list)
    target_column: Optional[str] = "Disease"
    target_classes: Optional[List[Any]] = Field(default_factory=lambda: [0, 1])
    class_balance: Optional[Dict[str, float]] = Field(default_factory=lambda: {"0": 0.62, "1": 0.38})
    missing_percentage: float = Field(default=5.0, ge=0.0, le=50.0)
    include_missing: bool = True
    random_seed: int = 42
    realistic_constraints: bool = True
    constraints: List[ConstraintConfig] = Field(default_factory=list)
    duplicate_percentage: float = Field(default=0.0, ge=0.0, le=20.0)
    noise_level: float = Field(default=0.1, ge=0.0, le=1.0)
    n_clusters: int = Field(default=3, ge=2, le=10)

    @field_validator("columns")
    @classmethod
    def validate_columns(cls, cols: List[ColumnConfig]) -> List[ColumnConfig]:
        if not cols:
            raise ValueError("Schema must contain at least one column")
        names = [c.name for c in cols]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate column names found in schema: {names}")
        return cols


# =====================================================================
# Preset Schemas
# =====================================================================

def get_healthcare_preset(task: str = "Classification", n_rows: int = 1000) -> DatasetSchema:
    """Healthcare preset matching the design mockup."""
    cols = [
        ColumnConfig(name="Patient_ID", col_type=ColumnType.ID, nullable=False),
        ColumnConfig(
            name="Age",
            col_type=ColumnType.NUMERIC,
            distribution=DistributionType.NORMAL,
            min_val=18.0,
            max_val=90.0,
            mean=48.3,
            std=16.2,
            decimals=0,
        ),
        ColumnConfig(
            name="Gender",
            col_type=ColumnType.CATEGORICAL,
            categories=["Male", "Female"],
            category_weights=[0.5, 0.5],
        ),
        ColumnConfig(
            name="BMI",
            col_type=ColumnType.NUMERIC,
            distribution=DistributionType.NORMAL,
            min_val=16.0,
            max_val=45.0,
            mean=27.5,
            std=5.1,
            decimals=1,
        ),
        ColumnConfig(
            name="Blood_Pressure",
            col_type=ColumnType.NUMERIC,
            distribution=DistributionType.NORMAL,
            min_val=90.0,
            max_val=180.0,
            mean=128.6,
            std=15.4,
            decimals=0,
        ),
        ColumnConfig(
            name="Cholesterol",
            col_type=ColumnType.NUMERIC,
            distribution=DistributionType.NORMAL,
            min_val=130.0,
            max_val=320.0,
            mean=210.0,
            std=35.0,
            decimals=0,
        ),
        ColumnConfig(
            name="Glucose",
            col_type=ColumnType.NUMERIC,
            distribution=DistributionType.NORMAL,
            min_val=70.0,
            max_val=260.0,
            mean=105.0,
            std=28.0,
            decimals=0,
        ),
        ColumnConfig(
            name="Smoking",
            col_type=ColumnType.CATEGORICAL,
            categories=["No", "Yes"],
            category_weights=[0.75, 0.25],
        ),
        ColumnConfig(
            name="Exercise_Hours",
            col_type=ColumnType.NUMERIC,
            distribution=DistributionType.UNIFORM,
            min_val=0.0,
            max_val=7.0,
            mean=3.0,
            std=2.0,
            decimals=0,
        ),
    ]

    target = "Disease"
    target_classes = [0, 1]
    class_balance = {"0": 0.624, "1": 0.376}

    if task == "Regression":
        target = "Recovery_Days"
        cols.append(
            ColumnConfig(
                name=target,
                col_type=ColumnType.NUMERIC,
                distribution=DistributionType.NORMAL,
                min_val=1.0,
                max_val=60.0,
                mean=14.0,
                std=8.0,
                is_target=True,
            )
        )
    elif task == "Classification":
        cols.append(
            ColumnConfig(
                name=target,
                col_type=ColumnType.NUMERIC,
                categories=["0", "1"],
                is_target=True,
                decimals=0,
            )
        )
    elif task == "Clustering":
        target = None

    constraints = [
        ConstraintConfig(
            rule="Blood_Pressure >= 90 and Blood_Pressure <= 180",
            description="Blood pressure must be within physiological limits",
        ),
        ConstraintConfig(
            rule="BMI >= 16.0 and BMI <= 45.0",
            description="BMI must be within realistic ranges",
        ),
    ]

    return DatasetSchema(
        domain="Healthcare",
        ml_task=task,
        n_rows=n_rows,
        columns=cols,
        target_column=target,
        target_classes=target_classes if task == "Classification" else None,
        class_balance=class_balance if task == "Classification" else None,
        missing_percentage=5.0,
        include_missing=True,
        random_seed=42,
        realistic_constraints=True,
        constraints=constraints,
    )


def get_banking_preset(task: str = "Classification", n_rows: int = 1000) -> DatasetSchema:
    """Banking and Finance preset."""
    cols = [
        ColumnConfig(name="Customer_ID", col_type=ColumnType.ID, nullable=False),
        ColumnConfig(
            name="Age",
            col_type=ColumnType.NUMERIC,
            distribution=DistributionType.NORMAL,
            min_val=21.0,
            max_val=72.0,
            mean=42.0,
            std=12.0,
            decimals=0,
        ),
        ColumnConfig(
            name="Employment_Type",
            col_type=ColumnType.CATEGORICAL,
            categories=["Salaried", "Self-Employed", "Business", "Unemployed"],
            category_weights=[0.55, 0.25, 0.15, 0.05],
        ),
        ColumnConfig(
            name="Annual_Income",
            col_type=ColumnType.NUMERIC,
            distribution=DistributionType.LOGNORMAL,
            min_val=15000.0,
            max_val=250000.0,
            mean=65000.0,
            std=25000.0,
            decimals=0,
        ),
        ColumnConfig(
            name="Credit_Score",
            col_type=ColumnType.NUMERIC,
            distribution=DistributionType.NORMAL,
            min_val=300.0,
            max_val=850.0,
            mean=675.0,
            std=68.0,
            decimals=0,
        ),
        ColumnConfig(
            name="Loan_Amount",
            col_type=ColumnType.NUMERIC,
            distribution=DistributionType.NORMAL,
            min_val=2000.0,
            max_val=100000.0,
            mean=25000.0,
            std=15000.0,
            decimals=0,
        ),
        ColumnConfig(
            name="Debt_To_Income",
            col_type=ColumnType.NUMERIC,
            distribution=DistributionType.UNIFORM,
            min_val=0.05,
            max_val=0.65,
            mean=0.32,
            std=0.15,
            decimals=2,
        ),
        ColumnConfig(
            name="Account_Balance",
            col_type=ColumnType.NUMERIC,
            distribution=DistributionType.LOGNORMAL,
            min_val=500.0,
            max_val=150000.0,
            mean=18500.0,
            std=12000.0,
            decimals=0,
        ),
        ColumnConfig(
            name="Past_Defaults",
            col_type=ColumnType.NUMERIC,
            distribution=DistributionType.POISSON,
            min_val=0.0,
            max_val=5.0,
            mean=0.4,
            std=0.7,
            decimals=0,
        ),
    ]

    target = "Loan_Defaulted"
    if task == "Regression":
        target = "Approved_Interest_Rate"
        cols.append(
            ColumnConfig(
                name=target,
                col_type=ColumnType.NUMERIC,
                min_val=3.5,
                max_val=24.0,
                mean=11.2,
                std=3.8,
                decimals=2,
                is_target=True,
            )
        )
    elif task == "Classification":
        cols.append(
            ColumnConfig(
                name=target,
                col_type=ColumnType.NUMERIC,
                categories=["0", "1"],
                is_target=True,
                decimals=0,
            )
        )
    elif task == "Clustering":
        target = None

    return DatasetSchema(
        domain="Banking and Finance",
        ml_task=task,
        n_rows=n_rows,
        columns=cols,
        target_column=target,
        target_classes=[0, 1] if task == "Classification" else None,
        class_balance={"0": 0.82, "1": 0.18} if task == "Classification" else None,
        missing_percentage=3.0,
        include_missing=True,
        random_seed=42,
    )


def get_ecommerce_preset(task: str = "Classification", n_rows: int = 1000) -> DatasetSchema:
    """E-commerce preset."""
    cols = [
        ColumnConfig(name="User_ID", col_type=ColumnType.ID, nullable=False),
        ColumnConfig(
            name="Age",
            col_type=ColumnType.NUMERIC,
            min_val=18.0,
            max_val=70.0,
            mean=35.0,
            std=11.0,
            decimals=0,
        ),
        ColumnConfig(
            name="Preferred_Category",
            col_type=ColumnType.CATEGORICAL,
            categories=["Electronics", "Fashion", "Home & Kitchen", "Beauty", "Sports"],
            category_weights=[0.3, 0.3, 0.18, 0.12, 0.1],
        ),
        ColumnConfig(
            name="Membership_Years",
            col_type=ColumnType.NUMERIC,
            min_val=0.0,
            max_val=8.0,
            mean=2.8,
            std=1.9,
            decimals=1,
        ),
        ColumnConfig(
            name="Total_Purchases",
            col_type=ColumnType.NUMERIC,
            distribution=DistributionType.POISSON,
            min_val=1.0,
            max_val=120.0,
            mean=22.0,
            std=14.0,
            decimals=0,
        ),
        ColumnConfig(
            name="Avg_Order_Value",
            col_type=ColumnType.NUMERIC,
            distribution=DistributionType.LOGNORMAL,
            min_val=12.0,
            max_val=480.0,
            mean=85.0,
            std=45.0,
            decimals=2,
        ),
        ColumnConfig(
            name="Days_Since_Last_Login",
            col_type=ColumnType.NUMERIC,
            min_val=0.0,
            max_val=180.0,
            mean=24.0,
            std=30.0,
            decimals=0,
        ),
        ColumnConfig(
            name="Discount_Usage_Ratio",
            col_type=ColumnType.NUMERIC,
            min_val=0.0,
            max_val=1.0,
            mean=0.35,
            std=0.22,
            decimals=2,
        ),
        ColumnConfig(
            name="Satisfaction_Score",
            col_type=ColumnType.NUMERIC,
            min_val=1.0,
            max_val=5.0,
            mean=3.9,
            std=0.9,
            decimals=0,
        ),
    ]

    target = "Churned"
    if task == "Regression":
        target = "Customer_Lifetime_Value"
        cols.append(
            ColumnConfig(
                name=target,
                col_type=ColumnType.NUMERIC,
                min_val=50.0,
                max_val=12000.0,
                mean=1850.0,
                std=950.0,
                decimals=2,
                is_target=True,
            )
        )
    elif task == "Classification":
        cols.append(
            ColumnConfig(
                name=target,
                col_type=ColumnType.NUMERIC,
                categories=["0", "1"],
                is_target=True,
                decimals=0,
            )
        )
    elif task == "Clustering":
        target = None

    return DatasetSchema(
        domain="E-commerce",
        ml_task=task,
        n_rows=n_rows,
        columns=cols,
        target_column=target,
        target_classes=[0, 1] if task == "Classification" else None,
        class_balance={"0": 0.74, "1": 0.26} if task == "Classification" else None,
        missing_percentage=4.0,
        include_missing=True,
        random_seed=42,
    )


def get_education_preset(task: str = "Classification", n_rows: int = 1000) -> DatasetSchema:
    """Education preset."""
    cols = [
        ColumnConfig(name="Student_ID", col_type=ColumnType.ID, nullable=False),
        ColumnConfig(
            name="Study_Hours_Per_Week",
            col_type=ColumnType.NUMERIC,
            min_val=2.0,
            max_val=40.0,
            mean=18.5,
            std=7.5,
            decimals=1,
        ),
        ColumnConfig(
            name="Attendance_Rate",
            col_type=ColumnType.NUMERIC,
            min_val=45.0,
            max_val=100.0,
            mean=82.0,
            std=11.0,
            decimals=1,
        ),
        ColumnConfig(
            name="Previous_GPA",
            col_type=ColumnType.NUMERIC,
            min_val=1.5,
            max_val=4.0,
            mean=3.1,
            std=0.5,
            decimals=2,
        ),
        ColumnConfig(
            name="Extracurricular_Activities",
            col_type=ColumnType.CATEGORICAL,
            categories=["None", "Sports", "Arts", "STEM Club", "Leadership"],
            category_weights=[0.25, 0.3, 0.15, 0.2, 0.1],
        ),
        ColumnConfig(
            name="Parental_Education",
            col_type=ColumnType.CATEGORICAL,
            categories=["High School", "Bachelor", "Master", "Doctorate"],
            category_weights=[0.35, 0.45, 0.15, 0.05],
        ),
        ColumnConfig(
            name="Internet_Access",
            col_type=ColumnType.BOOLEAN,
        ),
        ColumnConfig(
            name="Sleep_Hours",
            col_type=ColumnType.NUMERIC,
            min_val=4.0,
            max_val=10.0,
            mean=7.0,
            std=1.2,
            decimals=1,
        ),
        ColumnConfig(
            name="Tutoring_Sessions",
            col_type=ColumnType.NUMERIC,
            min_val=0.0,
            max_val=10.0,
            mean=1.5,
            std=1.8,
            decimals=0,
        ),
    ]

    target = "Exam_Passed"
    if task == "Regression":
        target = "Final_Exam_Score"
        cols.append(
            ColumnConfig(
                name=target,
                col_type=ColumnType.NUMERIC,
                min_val=30.0,
                max_val=100.0,
                mean=74.5,
                std=13.0,
                decimals=1,
                is_target=True,
            )
        )
    elif task == "Classification":
        cols.append(
            ColumnConfig(
                name=target,
                col_type=ColumnType.NUMERIC,
                categories=["0", "1"],
                is_target=True,
                decimals=0,
            )
        )
    elif task == "Clustering":
        target = None

    return DatasetSchema(
        domain="Education",
        ml_task=task,
        n_rows=n_rows,
        columns=cols,
        target_column=target,
        target_classes=[0, 1] if task == "Classification" else None,
        class_balance={"0": 0.35, "1": 0.65} if task == "Classification" else None,
        missing_percentage=2.0,
        include_missing=True,
        random_seed=42,
    )


def get_agriculture_preset(task: str = "Classification", n_rows: int = 1000) -> DatasetSchema:
    """Agriculture preset."""
    cols = [
        ColumnConfig(name="Field_ID", col_type=ColumnType.ID, nullable=False),
        ColumnConfig(
            name="Soil_Moisture",
            col_type=ColumnType.NUMERIC,
            min_val=10.0,
            max_val=75.0,
            mean=42.0,
            std=12.0,
            decimals=1,
        ),
        ColumnConfig(
            name="Soil_pH",
            col_type=ColumnType.NUMERIC,
            min_val=4.5,
            max_val=8.5,
            mean=6.5,
            std=0.8,
            decimals=2,
        ),
        ColumnConfig(
            name="Nitrogen_Content",
            col_type=ColumnType.NUMERIC,
            min_val=15.0,
            max_val=140.0,
            mean=75.0,
            std=25.0,
            decimals=1,
        ),
        ColumnConfig(
            name="Phosphorus_Content",
            col_type=ColumnType.NUMERIC,
            min_val=10.0,
            max_val=90.0,
            mean=48.0,
            std=18.0,
            decimals=1,
        ),
        ColumnConfig(
            name="Potassium_Content",
            col_type=ColumnType.NUMERIC,
            min_val=10.0,
            max_val=150.0,
            mean=65.0,
            std=22.0,
            decimals=1,
        ),
        ColumnConfig(
            name="Temperature_Celsius",
            col_type=ColumnType.NUMERIC,
            min_val=12.0,
            max_val=42.0,
            mean=26.5,
            std=5.5,
            decimals=1,
        ),
        ColumnConfig(
            name="Rainfall_mm",
            col_type=ColumnType.NUMERIC,
            min_val=30.0,
            max_val=320.0,
            mean=145.0,
            std=55.0,
            decimals=1,
        ),
        ColumnConfig(
            name="Irrigation_Type",
            col_type=ColumnType.CATEGORICAL,
            categories=["Drip", "Sprinkler", "Flood", "Rainfed"],
            category_weights=[0.35, 0.35, 0.15, 0.15],
        ),
    ]

    target = "Crop_Suitability"
    if task == "Regression":
        target = "Crop_Yield_Ton"
        cols.append(
            ColumnConfig(
                name=target,
                col_type=ColumnType.NUMERIC,
                min_val=1.2,
                max_val=15.0,
                mean=6.8,
                std=2.4,
                decimals=2,
                is_target=True,
            )
        )
    elif task == "Classification":
        cols.append(
            ColumnConfig(
                name=target,
                col_type=ColumnType.CATEGORICAL,
                categories=["Low", "Medium", "High"],
                is_target=True,
            )
        )
    elif task == "Clustering":
        target = None

    return DatasetSchema(
        domain="Agriculture",
        ml_task=task,
        n_rows=n_rows,
        columns=cols,
        target_column=target,
        target_classes=["Low", "Medium", "High"] if task == "Classification" else None,
        class_balance={"Low": 0.25, "Medium": 0.50, "High": 0.25} if task == "Classification" else None,
        missing_percentage=3.0,
        include_missing=True,
        random_seed=42,
    )


def get_general_preset(task: str = "Classification", n_rows: int = 1000) -> DatasetSchema:
    """General ML preset."""
    cols = [
        ColumnConfig(name="Record_ID", col_type=ColumnType.ID, nullable=False),
        ColumnConfig(
            name="Feature_1",
            col_type=ColumnType.NUMERIC,
            distribution=DistributionType.NORMAL,
            min_val=-10.0,
            max_val=10.0,
            mean=0.0,
            std=2.5,
            decimals=2,
        ),
        ColumnConfig(
            name="Feature_2",
            col_type=ColumnType.NUMERIC,
            distribution=DistributionType.NORMAL,
            min_val=0.0,
            max_val=100.0,
            mean=50.0,
            std=15.0,
            decimals=2,
        ),
        ColumnConfig(
            name="Feature_3",
            col_type=ColumnType.NUMERIC,
            distribution=DistributionType.UNIFORM,
            min_val=1.0,
            max_val=10.0,
            decimals=2,
        ),
        ColumnConfig(
            name="Category_Group",
            col_type=ColumnType.CATEGORICAL,
            categories=["Group_A", "Group_B", "Group_C"],
            category_weights=[0.4, 0.35, 0.25],
        ),
        ColumnConfig(
            name="Is_Active",
            col_type=ColumnType.BOOLEAN,
        ),
        ColumnConfig(
            name="Feature_4",
            col_type=ColumnType.NUMERIC,
            distribution=DistributionType.LOGNORMAL,
            min_val=1.0,
            max_val=50.0,
            mean=12.0,
            std=6.0,
            decimals=2,
        ),
        ColumnConfig(
            name="Feature_5",
            col_type=ColumnType.NUMERIC,
            distribution=DistributionType.NORMAL,
            min_val=10.0,
            max_val=90.0,
            mean=45.0,
            std=12.0,
            decimals=2,
        ),
        ColumnConfig(
            name="Sub_Category",
            col_type=ColumnType.CATEGORICAL,
            categories=["Type_1", "Type_2"],
            category_weights=[0.6, 0.4],
        ),
    ]

    target = "Target"
    if task == "Regression":
        target = "Target_Value"
        cols.append(
            ColumnConfig(
                name=target,
                col_type=ColumnType.NUMERIC,
                min_val=10.0,
                max_val=500.0,
                mean=150.0,
                std=45.0,
                decimals=2,
                is_target=True,
            )
        )
    elif task == "Classification":
        cols.append(
            ColumnConfig(
                name=target,
                col_type=ColumnType.NUMERIC,
                categories=["0", "1"],
                is_target=True,
                decimals=0,
            )
        )
    elif task == "Clustering":
        target = None

    return DatasetSchema(
        domain="General ML",
        ml_task=task,
        n_rows=n_rows,
        columns=cols,
        target_column=target,
        target_classes=[0, 1] if task == "Classification" else None,
        class_balance={"0": 0.50, "1": 0.50} if task == "Classification" else None,
        missing_percentage=5.0,
        include_missing=True,
        random_seed=42,
    )


DOMAIN_PRESETS = {
    "Healthcare": get_healthcare_preset,
    "Banking and Finance": get_banking_preset,
    "E-commerce": get_ecommerce_preset,
    "Education": get_education_preset,
    "Agriculture": get_agriculture_preset,
    "General ML": get_general_preset,
}


def build_schema_for_domain(
    domain: str,
    task: str = "Classification",
    n_rows: int = 1000,
    n_features: Optional[int] = None,
    target_var: Optional[str] = None,
    include_missing: bool = True,
    missing_pct: float = 5.0,
    realistic_constraints: bool = True,
    seed: int = 42,
    duplicate_pct: float = 0.0,
) -> DatasetSchema:
    """Build a complete DatasetSchema based on user choices."""
    builder = DOMAIN_PRESETS.get(domain, get_general_preset)
    schema = builder(task=task, n_rows=n_rows)
    schema.include_missing = include_missing
    schema.missing_percentage = missing_pct if include_missing else 0.0
    schema.realistic_constraints = realistic_constraints
    schema.random_seed = seed
    schema.duplicate_percentage = duplicate_pct

    # Adjust feature count if specified
    if n_features is not None and n_features > 0:
        id_cols = [c for c in schema.columns if c.col_type == ColumnType.ID]
        target_cols = [c for c in schema.columns if c.is_target]
        feature_cols = [c for c in schema.columns if c.col_type != ColumnType.ID and not c.is_target]

        target_count = max(1, n_features - len(id_cols) - len(target_cols))
        if len(feature_cols) > target_count:
            feature_cols = feature_cols[:target_count]
        elif len(feature_cols) < target_count:
            # Generate additional generic features
            needed = target_count - len(feature_cols)
            for i in range(1, needed + 1):
                col_name = f"Feature_{len(feature_cols) + i}"
                feature_cols.append(
                    ColumnConfig(
                        name=col_name,
                        col_type=ColumnType.NUMERIC,
                        distribution=DistributionType.NORMAL,
                        min_val=0.0,
                        max_val=100.0,
                        mean=50.0,
                        std=15.0,
                        decimals=1,
                    )
                )

        schema.columns = id_cols + feature_cols + target_cols

    # Override target variable name if provided
    if target_var and schema.target_column:
        old_target = schema.target_column
        schema.target_column = target_var
        for col in schema.columns:
            if col.name == old_target or col.is_target:
                col.name = target_var
                col.is_target = True

    return schema
