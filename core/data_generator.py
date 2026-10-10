"""Synthetic data generation engine.

Implements realistic, reproducible synthetic data generation using NumPy, Pandas,
and Faker with distribution modeling, multi-column relationships, realistic constraints,
and task-specific target generation (Classification, Regression, Clustering).
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from faker import Faker

from synthetic_dataset_generator.core.schema_builder import (
    ColumnConfig,
    ColumnType,
    DistributionType,
    DatasetSchema,
)


class DataGenerator:
    """Core synthetic data generator."""

    def __init__(self, schema: DatasetSchema):
        self.schema = schema
        self.seed = schema.random_seed
        self.rng = np.random.default_rng(self.seed)
        self.fake = Faker()
        Faker.seed(self.seed)

    def generate(self) -> pd.DataFrame:
        """Generate the complete synthetic DataFrame based on schema."""
        n_rows = self.schema.n_rows
        df = pd.DataFrame()

        # Step 1: Generate individual feature columns
        for col in self.schema.columns:
            if col.is_target:
                continue  # Target generated after features

            if col.col_type == ColumnType.ID:
                df[col.name] = np.arange(1, n_rows + 1)
            elif col.col_type == ColumnType.NUMERIC:
                df[col.name] = self._generate_numeric(col, n_rows)
            elif col.col_type == ColumnType.CATEGORICAL:
                df[col.name] = self._generate_categorical(col, n_rows)
            elif col.col_type == ColumnType.BOOLEAN:
                df[col.name] = self._generate_boolean(col, n_rows)
            elif col.col_type == ColumnType.DATETIME:
                df[col.name] = self._generate_datetime(col, n_rows)

        # Step 2: Inject domain-specific correlations if realistic_constraints is True
        if self.schema.realistic_constraints:
            df = self._apply_domain_correlations(df)

        # Step 3: Generate target variable based on task
        if self.schema.ml_task == "Classification" and self.schema.target_column:
            df[self.schema.target_column] = self._generate_classification_target(df)
        elif self.schema.ml_task == "Regression" and self.schema.target_column:
            df[self.schema.target_column] = self._generate_regression_target(df)
        elif self.schema.ml_task == "Clustering":
            df = self._structure_clustering_data(df)

        # Step 4: Apply explicit constraint checks and clipping
        df = self._enforce_constraints(df)

        # Step 5: Duplicate rows if requested
        if self.schema.duplicate_percentage > 0:
            df = self._inject_duplicates(df, self.schema.duplicate_percentage)

        # Step 6: Inject missing values if enabled
        if self.schema.include_missing and self.schema.missing_percentage > 0:
            df = self._inject_missing_values(df, self.schema.missing_percentage)

        return df

    # -------------------------------------------------------------------------
    # Column Type Generators
    # -------------------------------------------------------------------------

    def _generate_numeric(self, col: ColumnConfig, n: int) -> np.ndarray:
        """Generate numeric values according to distribution and bounds."""
        min_v = col.min_val if col.min_val is not None else 0.0
        max_v = col.max_val if col.max_val is not None else 100.0
        mean = col.mean if col.mean is not None else (min_v + max_v) / 2.0
        std = col.std if col.std is not None else (max_v - min_v) / 4.0

        dist = col.distribution or DistributionType.NORMAL

        if dist == DistributionType.NORMAL:
            data = self.rng.normal(mean, max(std, 1e-4), n)
        elif dist == DistributionType.UNIFORM:
            data = self.rng.uniform(min_v, max_v, n)
        elif dist == DistributionType.EXPONENTIAL:
            scale = max(mean - min_v, 1e-2)
            data = min_v + self.rng.exponential(scale, n)
        elif dist == DistributionType.LOGNORMAL:
            # Approximate lognormal params from mean and std
            var = std**2
            sigma2 = math.log(1.0 + var / max(mean**2, 1e-4))
            mu = math.log(max(mean, 1e-4)) - 0.5 * sigma2
            data = self.rng.lognormal(mu, math.sqrt(sigma2), n)
        elif dist == DistributionType.POISSON:
            lam = max(0.1, mean)
            data = self.rng.poisson(lam, n).astype(float)
        elif dist == DistributionType.BIMODAL:
            mode1 = self.rng.normal(mean - std * 0.8, std * 0.6, n // 2)
            mode2 = self.rng.normal(mean + std * 0.8, std * 0.6, n - (n // 2))
            data = np.concatenate([mode1, mode2])
            self.rng.shuffle(data)
        else:
            data = self.rng.normal(mean, std, n)

        # Clip within configured bounds
        data = np.clip(data, min_v, max_v)

        if col.decimals == 0:
            return np.round(data).astype(int)
        return np.round(data, col.decimals)

    def _generate_categorical(self, col: ColumnConfig, n: int) -> np.ndarray:
        """Generate categorical values using configured categories and weights."""
        cats = col.categories or ["Cat_A", "Cat_B"]
        weights = col.category_weights

        if weights:
            # Normalize weights
            w_sum = sum(weights)
            p = [w / w_sum for w in weights]
            if len(p) != len(cats):
                p = None
        else:
            p = None

        return self.rng.choice(cats, size=n, p=p)

    def _generate_boolean(self, col: ColumnConfig, n: int) -> np.ndarray:
        """Generate boolean values."""
        return self.rng.choice([True, False], size=n, p=[0.5, 0.5])

    def _generate_datetime(self, col: ColumnConfig, n: int) -> pd.Series:
        """Generate random dates between start and end date."""
        start = pd.Timestamp(col.date_start or "2023-01-01").value
        end = pd.Timestamp(col.date_end or "2024-12-31").value
        timestamps = self.rng.integers(start, end, size=n)
        return pd.to_datetime(timestamps).strftime("%Y-%m-%d")

    # -------------------------------------------------------------------------
    # Realistic Domain Relationships
    # -------------------------------------------------------------------------

    def _apply_domain_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add realistic correlations between features based on domain."""
        domain = self.schema.domain

        if domain == "Healthcare":
            # In Healthcare: BP & Cholesterol positively correlate with Age & BMI
            if "Age" in df.columns and "Blood_Pressure" in df.columns:
                age_factor = (df["Age"] - 48.0) / 30.0
                df["Blood_Pressure"] = np.clip(
                    df["Blood_Pressure"] + age_factor * 12.0 + self.rng.normal(0, 4, len(df)),
                    90, 180
                ).round().astype(int)

            if "BMI" in df.columns and "Blood_Pressure" in df.columns:
                bmi_factor = (df["BMI"] - 27.5) / 10.0
                df["Blood_Pressure"] = np.clip(
                    df["Blood_Pressure"] + bmi_factor * 8.0, 90, 180
                ).round().astype(int)

            if "Age" in df.columns and "Cholesterol" in df.columns:
                age_factor = (df["Age"] - 48.0) / 25.0
                df["Cholesterol"] = np.clip(
                    df["Cholesterol"] + age_factor * 18.0 + self.rng.normal(0, 8, len(df)),
                    130, 320
                ).round().astype(int)

            if "Smoking" in df.columns and "Blood_Pressure" in df.columns:
                is_smoker = (df["Smoking"] == "Yes").astype(int)
                df["Blood_Pressure"] = np.clip(
                    df["Blood_Pressure"] + is_smoker * 6.0, 90, 180
                ).round().astype(int)

        elif domain == "Banking and Finance":
            # Annual income & Credit score positively influence Loan Amount and Account Balance
            if "Annual_Income" in df.columns and "Account_Balance" in df.columns:
                df["Account_Balance"] = np.clip(
                    df["Account_Balance"] * 0.4 + df["Annual_Income"] * 0.25 + self.rng.normal(0, 3000, len(df)),
                    500, 250000
                ).round().astype(int)

            if "Credit_Score" in df.columns and "Debt_To_Income" in df.columns:
                # Lower credit score correlates with higher debt-to-income
                cs_norm = (850 - df["Credit_Score"]) / 550.0
                df["Debt_To_Income"] = np.clip(
                    df["Debt_To_Income"] * 0.6 + cs_norm * 0.35, 0.05, 0.70
                ).round(2)

        elif domain == "E-commerce":
            # Days since last login negatively correlates with total purchases
            if "Days_Since_Last_Login" in df.columns and "Total_Purchases" in df.columns:
                recency_penalty = (df["Days_Since_Last_Login"] / 180.0) * 10
                df["Total_Purchases"] = np.clip(
                    df["Total_Purchases"] - recency_penalty + self.rng.normal(0, 2, len(df)),
                    1, 150
                ).round().astype(int)

        elif domain == "Education":
            # Study hours and attendance strongly correlate with previous GPA
            if "Study_Hours_Per_Week" in df.columns and "Previous_GPA" in df.columns:
                boost = (df["Study_Hours_Per_Week"] / 40.0) * 0.8
                df["Previous_GPA"] = np.clip(
                    df["Previous_GPA"] * 0.7 + boost + 0.8, 1.5, 4.0
                ).round(2)

        elif domain == "Agriculture":
            # Soil moisture and rainfall correlate
            if "Rainfall_mm" in df.columns and "Soil_Moisture" in df.columns:
                df["Soil_Moisture"] = np.clip(
                    df["Soil_Moisture"] * 0.5 + (df["Rainfall_mm"] / 320.0) * 40.0,
                    10.0, 75.0
                ).round(1)

        return df

    # -------------------------------------------------------------------------
    # Target Generation (Classification, Regression, Clustering)
    # -------------------------------------------------------------------------

    def _generate_classification_target(self, df: pd.DataFrame) -> np.ndarray:
        """Generate realistic binary or multi-class target with controllable balance."""
        n = len(df)
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != self.schema.target_column and "ID" not in c.upper()]

        if not numeric_cols:
            # Fallback if no numeric columns
            classes = self.schema.target_classes or [0, 1]
            return self.rng.choice(classes, size=n)

        # Standardize numeric columns to build a realistic risk score / logit
        z_scores = np.zeros(n)
        for col in numeric_cols[:5]:  # use up to 5 key features
            vals = df[col].astype(float).values
            mean_v, std_v = np.nanmean(vals), np.nanstd(vals)
            if std_v > 1e-4:
                z_scores += (vals - mean_v) / std_v

        # Add controlled noise
        noise = self.rng.normal(0, self.schema.noise_level * 2.0, n)
        latent_score = z_scores + noise

        # Match class balance
        classes = self.schema.target_classes or [0, 1]

        if len(classes) == 2:
            # Binary classification (e.g. Disease 0 or 1)
            # Default target balance e.g. class 1 is ~37.6%
            target_1_ratio = 0.376
            if self.schema.class_balance:
                # Look for '1' or positive class
                target_1_ratio = self.schema.class_balance.get("1", self.schema.class_balance.get(1, 0.376))

            threshold = np.quantile(latent_score, 1.0 - target_1_ratio)
            target = np.where(latent_score >= threshold, classes[1], classes[0])
            return target
        else:
            # Multi-class
            quantiles = np.linspace(0, 1, len(classes) + 1)[1:-1]
            thresholds = [np.quantile(latent_score, q) for q in quantiles]
            target = np.select(
                [latent_score < thresholds[0]] +
                [np.logical_and(latent_score >= thresholds[i], latent_score < thresholds[i+1]) for i in range(len(thresholds)-1)] +
                [latent_score >= thresholds[-1]],
                classes
            )
            return target

    def _generate_regression_target(self, df: pd.DataFrame) -> np.ndarray:
        """Generate continuous regression target with meaningful feature relationships."""
        n = len(df)
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != self.schema.target_column and "ID" not in c.upper()]

        # Target column config
        target_cfg = next((c for c in self.schema.columns if c.is_target or c.name == self.schema.target_column), None)
        min_v = target_cfg.min_val if target_cfg and target_cfg.min_val is not None else 10.0
        max_v = target_cfg.max_val if target_cfg and target_cfg.max_val is not None else 100.0
        mean_v = target_cfg.mean if target_cfg and target_cfg.mean is not None else (min_v + max_v) / 2.0
        std_v = target_cfg.std if target_cfg and target_cfg.std is not None else (max_v - min_v) / 6.0
        decimals = target_cfg.decimals if target_cfg else 2

        if not numeric_cols:
            res = self.rng.normal(mean_v, std_v, n)
            return np.round(np.clip(res, min_v, max_v), decimals)

        # Weighted combination of up to 4 features + non-linear interaction + noise
        weights = [0.45, 0.30, 0.15, 0.10]
        y_latent = np.zeros(n)

        for i, col in enumerate(numeric_cols[:4]):
            vals = df[col].astype(float).values
            norm_vals = (vals - np.nanmean(vals)) / max(np.nanstd(vals), 1e-4)
            y_latent += weights[i] * norm_vals

        # Add non-linear effect if multiple features
        if len(numeric_cols) >= 2:
            f1 = (df[numeric_cols[0]].values - np.nanmean(df[numeric_cols[0]].values)) / max(np.nanstd(df[numeric_cols[0]].values), 1e-4)
            f2 = (df[numeric_cols[1]].values - np.nanmean(df[numeric_cols[1]].values)) / max(np.nanstd(df[numeric_cols[1]].values), 1e-4)
            y_latent += 0.15 * (f1 * f2)

        # Noise
        noise = self.rng.normal(0, self.schema.noise_level, n)
        y_norm = (y_latent + noise) / max(np.std(y_latent + noise), 1e-4)

        # Scale to target mean and std
        y_final = mean_v + y_norm * std_v
        y_final = np.clip(y_final, min_v, max_v)
        return np.round(y_final, decimals)

    def _structure_clustering_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inject cluster structures into numeric columns for clustering tasks."""
        k = self.schema.n_clusters
        n = len(df)
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and "ID" not in c.upper()]

        if not numeric_cols:
            return df

        # Assign cluster labels
        cluster_assignments = self.rng.integers(0, k, size=n)

        # For each numeric column, shift each cluster by distinct offsets
        for col in numeric_cols:
            if pd.api.types.is_integer_dtype(df[col]):
                df[col] = df[col].astype(float)
            std_v = np.std(df[col].values)
            offsets = np.linspace(-1.2 * std_v, 1.2 * std_v, k)
            for cluster_idx in range(k):
                mask = (cluster_assignments == cluster_idx)
                df.loc[mask, col] += offsets[cluster_idx]

        return df

    # -------------------------------------------------------------------------
    # Constraints and Post-processing
    # -------------------------------------------------------------------------

    def _enforce_constraints(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enforce configured bounds and physiological constraints."""
        for col in self.schema.columns:
            if col.name in df.columns and col.col_type == ColumnType.NUMERIC:
                if col.min_val is not None:
                    df[col.name] = df[col.name].clip(lower=col.min_val)
                if col.max_val is not None:
                    df[col.name] = df[col.name].clip(upper=col.max_val)
        return df

    def _inject_duplicates(self, df: pd.DataFrame, duplicate_pct: float) -> pd.DataFrame:
        """Inject exact duplicate rows if requested."""
        n_dups = int(len(df) * (duplicate_pct / 100.0))
        if n_dups > 0:
            dup_indices = self.rng.choice(len(df), size=n_dups, replace=True)
            duplicates = df.iloc[dup_indices].copy()
            df = pd.concat([df, duplicates], ignore_index=True)
        return df

    def _inject_missing_values(self, df: pd.DataFrame, missing_pct: float) -> pd.DataFrame:
        """Inject np.nan values at configured percentage across feature columns."""
        # Never inject nulls into ID column
        eligible_cols = [c for c in df.columns if "ID" not in c.upper()]
        if not eligible_cols:
            return df

        fraction = missing_pct / 100.0
        n_rows = len(df)

        for col in eligible_cols:
            # Find column config if nullable
            col_cfg = next((c for c in self.schema.columns if c.name == col), None)
            if col_cfg and not col_cfg.nullable:
                continue

            # Random mask
            mask = self.rng.random(n_rows) < fraction
            # Convert column to float if numeric integer or object if boolean to hold NaN
            if pd.api.types.is_integer_dtype(df[col]):
                df[col] = df[col].astype(float)
            elif pd.api.types.is_bool_dtype(df[col]):
                df[col] = df[col].astype(object)
            df.loc[mask, col] = np.nan

        return df
