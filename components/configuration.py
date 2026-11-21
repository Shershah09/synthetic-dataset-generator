"""Dataset Configuration Panel Component.

Renders the configuration card matching the UI design:
- Domain / Use Case
- ML Task Type
- Dataset Size (Rows)
- Number of Features (Columns)
- Target Variable
- Include Missing Values & Percentage
- Add Realistic Constraints
- Advanced Options expander (Seed, noise, class balance, clusters, duplicates)
- Generate & Reset actions
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import streamlit as st

try:
    from core.schema_builder import (
        DatasetSchema,
        build_schema_for_domain,
    )
except ImportError:
    from synthetic_dataset_generator.core.schema_builder import (
        DatasetSchema,
        build_schema_for_domain,
    )


def render_configuration_panel() -> Tuple[bool, Optional[DatasetSchema]]:
    """Render the Dataset Configuration card.

    Returns:
        Tuple of (generate_clicked, schema)
    """
    st.markdown("""
        <div class="card-header-row">
            <div class="card-title">
                <span class="icon">⚙️</span> Dataset Configuration
            </div>
        </div>
    """, unsafe_allow_html=True)

    header_col1, header_col2 = st.columns([4, 1])
    with header_col2:
        if st.button("Reset", key="reset_config_btn", use_container_width=True):
            st.session_state["domain"] = "Healthcare"
            st.session_state["ml_task"] = "Classification"
            st.session_state["n_rows"] = 1000
            st.session_state["n_features"] = 10
            st.session_state["include_missing_opt"] = "Yes (Custom %)"
            st.session_state["missing_pct"] = 5.0
            st.session_state["realistic_constraints"] = "Yes"
            st.session_state["random_seed"] = 42
            st.session_state["noise_level"] = 0.1
            st.session_state["duplicate_pct"] = 0.0
            st.session_state["current_dataset"] = None
            st.session_state["quality_report"] = None
            st.rerun()

    # 1. Domain / Use Case
    domain_options = [
        "Healthcare",
        "Banking and Finance",
        "E-commerce",
        "Education",
        "Agriculture",
        "General ML",
    ]
    cur_domain = st.session_state.get("domain", "Healthcare")
    domain_idx = domain_options.index(cur_domain) if cur_domain in domain_options else 0
    domain = st.selectbox(
        "Domain / Use Case",
        domain_options,
        index=domain_idx,
        key="config_domain_select",
        help="Select business or scientific domain preset",
    )
    st.session_state["domain"] = domain

    # 2. ML Task Type
    task_options = ["Classification", "Regression", "Clustering", "General-purpose"]
    cur_task = st.session_state.get("ml_task", "Classification")
    task_idx = task_options.index(cur_task) if cur_task in task_options else 0
    ml_task = st.selectbox(
        "ML Task Type",
        task_options,
        index=task_idx,
        key="config_task_select",
        help="Target structure and relationship modeling",
    )
    st.session_state["ml_task"] = ml_task

    # 3. Dataset Size (Rows)
    size_options = [100, 250, 500, 1000, 2500, 5000, 10000]
    cur_rows = st.session_state.get("n_rows", 1000)
    size_idx = size_options.index(cur_rows) if cur_rows in size_options else 3
    n_rows = st.selectbox(
        "Dataset Size (Rows)",
        size_options,
        index=size_idx,
        key="config_size_select",
        help="Number of artificial records to synthesize",
    )
    st.session_state["n_rows"] = n_rows

    # 4. Number of Features (Columns)
    cur_features = st.session_state.get("n_features", 10)
    n_features = st.number_input(
        "Number of Features (Columns)",
        min_value=3,
        max_value=30,
        value=cur_features,
        step=1,
        key="config_features_input",
        help="Total columns to synthesize including target",
    )
    st.session_state["n_features"] = n_features

    # 5. Target Variable
    target_presets = {
        "Healthcare": ["Disease", "Diagnosis", "Readmission_Risk"],
        "Banking and Finance": ["Loan_Defaulted", "Approved", "Fraud_Detected"],
        "E-commerce": ["Churned", "High_Spender", "Repurchase"],
        "Education": ["Exam_Passed", "Dropout_Risk", "Honor_Roll"],
        "Agriculture": ["Crop_Suitability", "Pest_Infestation", "Optimal_Harvest"],
        "General ML": ["Target", "Class_Label", "Outcome"],
    }
    avail_targets = target_presets.get(domain, ["Target"])
    cur_target = st.session_state.get("target_var", avail_targets[0])
    target_idx = avail_targets.index(cur_target) if cur_target in avail_targets else 0
    target_var = st.selectbox(
        "Target Variable",
        avail_targets,
        index=target_idx,
        disabled=(ml_task == "Clustering" or ml_task == "General-purpose"),
        key="config_target_select",
        help="Supervised learning target label",
    )
    st.session_state["target_var"] = target_var

    # 6. Include Missing Values
    missing_opts = ["Yes (Custom %)", "No", "Yes (5% Random)", "Yes (10% Random)"]
    cur_missing_opt = st.session_state.get("include_missing_opt", "Yes (Custom %)")
    missing_opt_idx = missing_opts.index(cur_missing_opt) if cur_missing_opt in missing_opts else 0
    missing_opt = st.selectbox(
        "Include Missing Values",
        missing_opts,
        index=missing_opt_idx,
        key="config_missing_opt",
    )
    st.session_state["include_missing_opt"] = missing_opt

    include_missing = missing_opt != "No"
    missing_pct = 5.0
    if missing_opt == "Yes (Custom %)":
        missing_pct = st.number_input(
            "Missing Value Percentage (%)",
            min_value=0.0,
            max_value=40.0,
            value=float(st.session_state.get("missing_pct", 5.0)),
            step=1.0,
            key="config_missing_pct_input",
        )
    elif missing_opt == "Yes (5% Random)":
        missing_pct = 5.0
    elif missing_opt == "Yes (10% Random)":
        missing_pct = 10.0
    else:
        missing_pct = 0.0
    st.session_state["missing_pct"] = missing_pct

    # 7. Add Realistic Constraints
    constraints_opts = ["Yes", "No"]
    cur_constraints = st.session_state.get("realistic_constraints", "Yes")
    constraints_idx = 0 if cur_constraints == "Yes" else 1
    realistic_constraints_val = st.selectbox(
        "Add Realistic Constraints",
        constraints_opts,
        index=constraints_idx,
        key="config_constraints_select",
        help="Model physiological, domain correlations and numerical bounds",
    )
    st.session_state["realistic_constraints"] = realistic_constraints_val
    realistic_constraints = (realistic_constraints_val == "Yes")

    # 8. Advanced Options Expander
    with st.expander("🛠 Advanced Options", expanded=False):
        seed = st.number_input(
            "Random Seed (Reproducibility)",
            min_value=0,
            max_value=999999,
            value=int(st.session_state.get("random_seed", 42)),
            step=1,
            key="adv_seed_input",
            help="Ensures identical dataset output across runs",
        )
        st.session_state["random_seed"] = seed

        noise_level = st.slider(
            "Noise Level (Controlled Realism)",
            min_value=0.0,
            max_value=0.5,
            value=float(st.session_state.get("noise_level", 0.1)),
            step=0.05,
            key="adv_noise_slider",
            help="Statistical noise in feature relationships and labels",
        )
        st.session_state["noise_level"] = noise_level

        class_balance_val = 37.6
        if ml_task == "Classification":
            class_balance_val = st.slider(
                "Positive Class Balance (%)",
                min_value=10.0,
                max_value=90.0,
                value=float(st.session_state.get("class_balance_pct", 37.6)),
                step=1.0,
                key="adv_class_balance_slider",
                help="Proportion of positive class (Disease = 1)",
            )
            st.session_state["class_balance_pct"] = class_balance_val

        duplicate_pct = st.number_input(
            "Duplicate Rows (%)",
            min_value=0.0,
            max_value=15.0,
            value=float(st.session_state.get("duplicate_pct", 0.0)),
            step=0.5,
            key="adv_duplicate_input",
            help="Simulate imperfect data collection with duplicate rows",
        )
        st.session_state["duplicate_pct"] = duplicate_pct

        n_clusters = 3
        if ml_task == "Clustering":
            n_clusters = st.slider(
                "Number of Clusters",
                min_value=2,
                max_value=8,
                value=3,
                key="adv_clusters_slider",
            )

    # Primary Action Button
    generate_btn = st.button(
        "▶ Generate Dataset",
        type="primary",
        use_container_width=True,
        key="btn_generate_main",
    )

    # Build schema
    schema = build_schema_for_domain(
        domain=domain,
        task="Clustering" if ml_task == "Clustering" else ("Regression" if ml_task == "Regression" else "Classification"),
        n_rows=n_rows,
        n_features=n_features,
        target_var=target_var,
        include_missing=include_missing,
        missing_pct=missing_pct,
        realistic_constraints=realistic_constraints,
        seed=seed,
        duplicate_pct=duplicate_pct,
    )
    schema.noise_level = noise_level
    if ml_task == "Classification":
        pos_ratio = class_balance_val / 100.0
        schema.class_balance = {"0": 1.0 - pos_ratio, "1": pos_ratio}
    if ml_task == "Clustering":
        schema.n_clusters = n_clusters

    return generate_btn, schema
