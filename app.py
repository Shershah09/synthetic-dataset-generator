"""Synthetic Dataset Generator - Streamlit Web Application.

A full-stack, modular web application for configuring, generating, validating,
and downloading realistic synthetic datasets for ML/DL and research projects.
"""

from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np

# Ensure root package is in sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir.parent) not in sys.path:
    sys.path.insert(0, str(root_dir.parent))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from synthetic_dataset_generator.core.schema_builder import (
    DatasetSchema,
    build_schema_for_domain,
    DOMAIN_PRESETS,
)
from synthetic_dataset_generator.core.data_generator import DataGenerator
from synthetic_dataset_generator.core.validator import DataValidator, QualityReport
from synthetic_dataset_generator.core.exporter import ExportService
from synthetic_dataset_generator.components.configuration import render_configuration_panel
from synthetic_dataset_generator.components.preview import render_preview_panel
from synthetic_dataset_generator.components.quality_dashboard import render_analytics_and_downloads

# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title="Synthetic Dataset Generator",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Custom CSS for Modern SaaS UI (matching reference screenshot)
# =============================================================================
CUSTOM_CSS = """
<style>
/* Main Container & Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

/* Base App Background */
.stApp {
    background-color: #f8fafc;
    color: #0f172a;
}

/* Sidebar Styling: Dark Navy Blue Theme */
[data-testid="stSidebar"] {
    background-color: #0f172a !important;
    border-right: 1px solid #1e293b;
    padding-top: 1rem;
}

[data-testid="stSidebar"] * {
    color: #cbd5e1;
}

[data-testid="stSidebar"] hr {
    border-color: #1e293b;
}

/* Sidebar Logo Header */
.sidebar-brand-container {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0.5rem 0.25rem 1.25rem 0.25rem;
    border-bottom: 1px solid #1e293b;
    margin-bottom: 1.25rem;
}
.sidebar-brand-icon {
    font-size: 2rem;
    line-height: 1;
}
.sidebar-brand-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #ffffff !important;
    letter-spacing: -0.02em;
}
.sidebar-brand-sub {
    font-size: 0.72rem;
    color: #94a3b8 !important;
    font-weight: 500;
}

/* Sidebar Nav Buttons */
.nav-item-btn {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 9px 14px;
    border-radius: 8px;
    font-size: 0.9rem;
    font-weight: 500;
    margin-bottom: 4px;
    cursor: pointer;
    transition: all 0.15s ease-in-out;
    text-decoration: none;
    color: #94a3b8;
}
.nav-item-btn:hover {
    background-color: #1e293b;
    color: #ffffff;
}
.nav-item-btn.active {
    background-color: #2563eb !important;
    color: #ffffff !important;
    font-weight: 600;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35);
}

/* Sidebar Bottom AI Card */
.sidebar-ai-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 14px;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 2rem;
}
.sidebar-ai-icon {
    background: #2563eb;
    color: white;
    width: 38px;
    height: 38px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.25rem;
    flex-shrink: 0;
}
.sidebar-ai-text {
    font-size: 0.75rem;
    color: #94a3b8;
    line-height: 1.3;
}
.sidebar-ai-text b {
    color: #f8fafc;
    font-size: 0.82rem;
}

/* Top Navigation Bar */
.top-header-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0 1.25rem 0;
    border-bottom: 1px solid #e2e8f0;
    margin-bottom: 1.25rem;
}
.header-title-block h1 {
    font-size: 1.6rem;
    font-weight: 700;
    color: #0f172a;
    margin: 0;
    padding: 0;
    letter-spacing: -0.02em;
}
.header-title-block p {
    font-size: 0.85rem;
    color: #64748b;
    margin: 3px 0 0 0;
}
.top-nav-links {
    display: flex;
    align-items: center;
    gap: 8px;
}
.top-pill {
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 500;
    text-decoration: none;
    color: #475569;
    border: 1px solid transparent;
}
.top-pill.active {
    background-color: #eff6ff;
    color: #2563eb;
    border-color: #bfdbfe;
    font-weight: 600;
}

/* Horizontal Stepper */
.stepper-container {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 12px 24px;
    margin-bottom: 1.5rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
}
.step-item {
    display: flex;
    align-items: center;
    gap: 10px;
}
.step-circle {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.82rem;
    font-weight: 600;
    background: #f1f5f9;
    color: #64748b;
}
.step-circle.active {
    background: #2563eb;
    color: #ffffff;
    box-shadow: 0 2px 6px rgba(37, 99, 235, 0.3);
}
.step-circle.completed {
    background: #10b981;
    color: #ffffff;
}
.step-label {
    font-size: 0.85rem;
    font-weight: 600;
    color: #334155;
}
.step-line {
    flex: 1;
    height: 2px;
    background: #e2e8f0;
    margin: 0 16px;
}
.step-line.active {
    background: #2563eb;
}

/* Card Styling */
.card-header-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.75rem;
    padding-bottom: 0.35rem;
}
.card-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #0f172a;
    display: flex;
    align-items: center;
    gap: 8px;
}
.badges-row {
    display: flex;
    gap: 8px;
}
.badge {
    padding: 3px 10px;
    border-radius: 16px;
    font-size: 0.72rem;
    font-weight: 600;
}
.badge-blue {
    background-color: #eff6ff;
    color: #2563eb;
    border: 1px solid #bfdbfe;
}
.badge-green {
    background-color: #ecfdf5;
    color: #059669;
    border: 1px solid #a7f3d0;
}

/* Custom Primary Button Styling */
div.stButton > button[kind="primary"] {
    background-color: #2563eb !important;
    color: white !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.55rem 1rem !important;
    border: none !important;
    box-shadow: 0 2px 8px rgba(37, 99, 235, 0.25) !important;
    transition: all 0.15s ease !important;
}
div.stButton > button[kind="primary"]:hover {
    background-color: #1d4ed8 !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4) !important;
}

/* Download Buttons Styling */
div[data-testid="stDownloadButton"] > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    border: none !important;
    transition: all 0.15s ease !important;
    color: white !important;
}
#dl_btn_csv > button, div[data-testid="stDownloadButton"]:nth-of-type(1) > button {
    background-color: #10b981 !important;
}
#dl_btn_excel > button, div[data-testid="stDownloadButton"]:nth-of-type(2) > button {
    background-color: #2563eb !important;
}
#dl_btn_json > button, div[data-testid="stDownloadButton"]:nth-of-type(3) > button {
    background-color: #8b5cf6 !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# Session State Initialization
# =============================================================================
def init_session_state():
    """Initialize default state with Healthcare preset for instant preview."""
    if "current_nav" not in st.session_state:
        st.session_state["current_nav"] = "Generate Dataset"

    if "current_dataset" not in st.session_state or st.session_state["current_dataset"] is None:
        # Generate initial default dataset matching UI mockup
        default_schema = build_schema_for_domain("Healthcare", task="Classification", n_rows=1000)
        generator = DataGenerator(default_schema)
        raw_df = generator.generate()
        validator = DataValidator(default_schema)
        clean_df, report = validator.validate(raw_df, auto_repair=True)

        st.session_state["current_dataset"] = clean_df
        st.session_state["dataset_schema"] = default_schema
        st.session_state["quality_report"] = report
        st.session_state["domain"] = "Healthcare"
        st.session_state["ml_task"] = "Classification"
        st.session_state["n_rows"] = 1000
        st.session_state["n_features"] = 10
        st.session_state["target_var"] = "Disease"
        st.session_state["include_missing_opt"] = "Yes (Custom %)"
        st.session_state["missing_pct"] = 5.0
        st.session_state["realistic_constraints"] = "Yes"
        st.session_state["random_seed"] = 42

    if "saved_datasets" not in st.session_state:
        st.session_state["saved_datasets"] = [
            {
                "id": "ds_healthcare_01",
                "name": "Healthcare Disease Classification (1,000 rows)",
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "domain": "Healthcare",
                "task": "Classification",
                "rows": 1000,
                "cols": 10,
                "score": 98,
                "dataframe": st.session_state["current_dataset"],
                "schema": st.session_state["dataset_schema"],
                "report": st.session_state["quality_report"],
            }
        ]


init_session_state()


# =============================================================================
# Sidebar Navigation
# =============================================================================
with st.sidebar:
    st.markdown("""
        <div class="sidebar-brand-container">
            <div class="sidebar-brand-icon">🗄️</div>
            <div>
                <div class="sidebar-brand-title">Synthetic</div>
                <div class="sidebar-brand-sub">Dataset Generator</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    nav_options = [
        ("Generate Dataset", "📊"),
        ("Saved Datasets", "📁"),
        ("Templates", "📄"),
        ("Documentation", "📖"),
        ("Learn & Examples", "🎓"),
        ("Settings", "⚙️"),
    ]

    for label, icon in nav_options:
        is_active = (st.session_state["current_nav"] == label)
        btn_type = "primary" if is_active else "secondary"
        if st.button(f"{icon}  {label}", key=f"nav_btn_{label}", use_container_width=True, type=btn_type):
            st.session_state["current_nav"] = label
            st.rerun()

    # Bottom AI integration card
    st.markdown("""
        <div class="sidebar-ai-card">
            <div class="sidebar-ai-icon">🤖</div>
            <div class="sidebar-ai-text">
                Powered by<br>
                <b>Generative AI<br>& Agentic AI</b>
            </div>
        </div>
    """, unsafe_allow_html=True)


# =============================================================================
# Top Header Bar & Workflow Stepper
# =============================================================================
def render_header():
    """Render top navbar and horizontal 4-step workflow indicator."""
    st.markdown("""
        <div class="top-header-container">
            <div class="header-title-block">
                <h1>Dataset Generator</h1>
                <p>Create realistic synthetic datasets for ML, DL and research projects</p>
            </div>
            <div class="top-nav-links">
                <span class="top-pill active">🏠 Home</span>
                <span class="top-pill">About</span>
                <span class="top-pill">⚙️ How it Works</span>
                <span class="top-pill">📄 Examples</span>
                <span style="font-size: 1.1rem; margin-left: 6px; cursor: pointer;">🌙</span>
                <span style="font-size: 1.25rem; margin-left: 6px; cursor: pointer;">👤</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    has_data = st.session_state.get("current_dataset") is not None
    st.markdown(f"""
        <div class="stepper-container">
            <div class="step-item">
                <div class="step-circle {'completed' if has_data else 'active'}">1</div>
                <div class="step-label">Configure Dataset</div>
            </div>
            <div class="step-line active"></div>
            <div class="step-item">
                <div class="step-circle {'completed' if has_data else 'active'}">2</div>
                <div class="step-label">Generate Synthetic Data</div>
            </div>
            <div class="step-line {'active' if has_data else ''}"></div>
            <div class="step-item">
                <div class="step-circle {'active' if has_data else ''}">3</div>
                <div class="step-label">Preview & Analyse</div>
            </div>
            <div class="step-line {'active' if has_data else ''}"></div>
            <div class="step-item">
                <div class="step-circle {'active' if has_data else ''}">4</div>
                <div class="step-label">Download Dataset</div>
            </div>
        </div>
    """, unsafe_allow_html=True)


# =============================================================================
# Page Views
# =============================================================================

def view_generate_dataset():
    """Main Dashboard View: Two-column layout matching reference screenshot."""
    render_header()

    col_config, col_main = st.columns([1.0, 1.8], gap="medium")

    # LEFT COLUMN: Dataset Configuration
    with col_config:
        generate_clicked, schema = render_configuration_panel()

    # Action Trigger: Generate Dataset
    if generate_clicked and schema:
        with st.spinner("⚡ Synthesizing realistic records, applying domain constraints & auditing quality..."):
            try:
                # Core Engine Execution
                generator = DataGenerator(schema)
                raw_df = generator.generate()

                # Validation & Auto-repair
                validator = DataValidator(schema)
                clean_df, report = validator.validate(raw_df, auto_repair=True)

                # Update State
                st.session_state["current_dataset"] = clean_df
                st.session_state["dataset_schema"] = schema
                st.session_state["quality_report"] = report

                # Save to session history
                new_saved = {
                    "id": f"ds_{len(st.session_state['saved_datasets']) + 1:02d}",
                    "name": f"{schema.domain} {schema.ml_task} ({len(clean_df):,} rows)",
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "domain": schema.domain,
                    "task": schema.ml_task,
                    "rows": len(clean_df),
                    "cols": len(clean_df.columns),
                    "score": report.overall_score,
                    "dataframe": clean_df,
                    "schema": schema,
                    "report": report,
                }
                st.session_state["saved_datasets"].append(new_saved)
                st.success(f"✨ Successfully generated {len(clean_df):,} records! Quality Score: {report.overall_score}/100")
            except Exception as e:
                st.error(f"❌ Generation error: {str(e)}")

    # RIGHT COLUMN: Preview & Analytics
    with col_main:
        current_df = st.session_state.get("current_dataset")
        current_schema = st.session_state.get("dataset_schema")
        current_report = st.session_state.get("quality_report")

        # 1. Dataset Preview Table
        render_preview_panel(current_df)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # 2. Analytics Row & Download Panel
        render_analytics_and_downloads(current_df, current_schema, current_report)


def view_saved_datasets():
    """Saved Datasets View: Manage datasets within the active session."""
    st.markdown("## 📁 Saved Datasets (Current Session)")
    st.caption("Inspect, restore, or export datasets created during your active browser session.")

    saved_list = st.session_state.get("saved_datasets", [])
    if not saved_list:
        st.info("No datasets saved yet. Generate a dataset from the main dashboard!")
        return

    for idx, item in enumerate(reversed(saved_list)):
        with st.container():
            c1, c2, c3, c4, c5 = st.columns([3, 1.5, 1.2, 1.5, 1.2])
            c1.markdown(f"**{item['name']}**<br><span style='color: #64748b; font-size: 0.8rem;'>{item['created_at']}</span>", unsafe_allow_html=True)
            c2.markdown(f"**Domain:** {item['domain']}<br>**Task:** {item['task']}", unsafe_allow_html=True)
            c3.markdown(f"**Shape:** {item['rows']}×{item['cols']}<br>**Quality:** `{item['score']}/100`", unsafe_allow_html=True)

            with c4:
                if st.button(f"Load into Preview", key=f"load_ds_{idx}"):
                    st.session_state["current_dataset"] = item["dataframe"]
                    st.session_state["dataset_schema"] = item["schema"]
                    st.session_state["quality_report"] = item["report"]
                    st.session_state["current_nav"] = "Generate Dataset"
                    st.rerun()

            with c5:
                csv_bytes, _, csv_name = ExportService.to_csv(item["dataframe"], item["domain"], item["task"])
                st.download_button(
                    "Download",
                    data=csv_bytes,
                    file_name=csv_name,
                    mime="text/csv",
                    key=f"dl_saved_{idx}",
                )
            st.divider()


def view_templates():
    """Templates View: Pre-configured domain schemas ready for 1-click loading."""
    st.markdown("## 📄 Dataset Templates")
    st.caption("Pre-configured industry standard schemas optimized for ML benchmarks.")

    templates = [
        {
            "domain": "Healthcare",
            "task": "Classification",
            "name": "Clinical Heart & Metabolic Disease Screening",
            "desc": "10 clinical features (Age, Blood Pressure, Cholesterol, Glucose, BMI, Smoking) predicting disease incidence.",
        },
        {
            "domain": "Banking and Finance",
            "task": "Classification",
            "name": "Consumer Credit Risk & Default Prediction",
            "desc": "Credit score, debt-to-income, loan amount, and repayment history with imbalanced target.",
        },
        {
            "domain": "E-commerce",
            "task": "Classification",
            "name": "Customer Churn & Retention Analytics",
            "desc": "Activity recency, transaction frequency, lifetime value, and satisfaction metrics.",
        },
        {
            "domain": "Education",
            "task": "Regression",
            "name": "Academic Performance & Exam Score Forecasting",
            "desc": "Study hours, attendance rates, sleep duration, and previous GPA predicting continuous exam marks.",
        },
        {
            "domain": "Agriculture",
            "task": "Regression",
            "name": "Precision Soil & Crop Yield Estimation",
            "desc": "Soil moisture, pH, N-P-K mineral levels, rainfall, and temperature predicting yield tons per hectare.",
        },
        {
            "domain": "General ML",
            "task": "Clustering",
            "name": "High-Dimensional Unsupervised Clustering Benchmark",
            "desc": "Multi-modal Gaussian distributions with distinct cluster centroids for K-Means and DBSCAN testing.",
        },
    ]

    t_cols = st.columns(2)
    for idx, tmpl in enumerate(templates):
        with t_cols[idx % 2]:
            st.markdown(f"""
                <div style="background: white; border: 1px solid #e2e8f0; border-radius: 10px; padding: 18px; margin-bottom: 16px;">
                    <div style="font-size: 0.75rem; font-weight: 600; color: #2563eb; text-transform: uppercase;">{tmpl['domain']} • {tmpl['task']}</div>
                    <div style="font-size: 1.1rem; font-weight: 700; color: #0f172a; margin: 4px 0 8px 0;">{tmpl['name']}</div>
                    <div style="font-size: 0.85rem; color: #64748b; margin-bottom: 14px;">{tmpl['desc']}</div>
                </div>
            """, unsafe_allow_html=True)
            if st.button(f"Load Template", key=f"btn_load_tmpl_{idx}"):
                st.session_state["domain"] = tmpl["domain"]
                st.session_state["ml_task"] = tmpl["task"]
                st.session_state["current_nav"] = "Generate Dataset"
                st.rerun()


def view_documentation():
    """Documentation View: System architecture, generation mechanics, and validation rules."""
    st.markdown("## 📖 System Documentation")
    st.markdown("""
### Architecture & Design Philosophy
The **Synthetic Dataset Generator** provides realistic, controllable artificial data for Machine Learning and Deep Learning experimentation without privacy or regulatory restrictions.

#### 1. Core Generation Engine
- **Reproducibility:** Seeded pseudo-random number generator (`numpy.random.default_rng`) guarantees bitwise identical datasets across runs.
- **Statistical Distributions:** Features are generated using parametric distributions (Gaussian, Uniform, Lognormal, Poisson, Bimodal).
- **Physiological & Business Constraints:** Enforces hard boundary clipping and natural domain interactions (e.g. Blood Pressure rising with Age and BMI).
- **Target Modeling:**
  - **Classification:** Latent logistic linear combination with controlled noise and quantile-based thresholding for exact class balance.
  - **Regression:** Weighted multi-feature relationships with non-linear interaction terms and configurable Gaussian residuals.
  - **Clustering:** Multi-centroid space structuring with cluster-specific mean shifts.

#### 2. Data Validation & Quality Index
Every generated dataset undergoes automated auditing:
1. **Structural Checks:** Verifies shape, non-empty columns, and type casting.
2. **Bounds & Domain Integrity:** Checks that numeric values respect configured bounds and categorical values conform to domain dictionaries.
3. **Missing Value Verification:** Confirms random missing value masks match user-specified percentages.
4. **Infinite/NaN Auditing:** Automatically repairs unexpected infinites by clipping to column medians.

#### 3. Export Capabilities
Exports are streamed directly in-memory to prevent persistent server disk accumulation:
- **CSV:** UTF-8 encoded with standard comma separators.
- **Excel (.xlsx):** Multi-column workbook generated via OpenPyXL.
- **JSON:** Record-oriented structured array for REST API ingestion.
    """)


def view_learn_and_examples():
    """Learn & Examples View: Code snippets for loading synthetic data in ML frameworks."""
    st.markdown("## 🎓 Learn & Code Examples")
    st.markdown("Integrate the exported synthetic datasets directly into your ML/DL workflows.")

    tab1, tab2, tab3 = st.tabs(["Scikit-Learn Classification", "PyTorch Custom Dataset", "Pandas EDA Quickstart"])

    with tab1:
        st.code("""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. Load generated synthetic dataset
df = pd.read_csv("synthetic_healthcare_classification.csv")

# 2. Feature selection and target separation
X = df.drop(columns=["Patient_ID", "Disease"])
X = pd.get_dummies(X, drop_first=True)  # Encode categoricals
y = df["Disease"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train.fillna(X_train.median()), y_train)

# 5. Evaluate
y_pred = clf.predict(X_test.fillna(X_train.median()))
print(classification_report(y_test, y_pred))
        """, language="python")

    with tab2:
        st.code("""
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class SyntheticMLDataset(Dataset):
    def __init__(self, csv_file, target_col="Disease"):
        df = pd.read_csv(csv_file).dropna()
        feature_cols = [c for c in df.select_dtypes(include=['float64', 'int64']).columns if c != target_col and 'ID' not in c]
        self.X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.y = torch.tensor(df[target_col].values, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = SyntheticMLDataset("synthetic_healthcare_classification.csv")
loader = DataLoader(dataset, batch_size=32, shuffle=True)
for batch_x, batch_y in loader:
    print(f"Batch X shape: {batch_x.shape}, Batch y shape: {batch_y.shape}")
    break
        """, language="python")

    with tab3:
        st.code("""
import pandas as pd

df = pd.read_csv("synthetic_healthcare_classification.csv")
print("Dataset Info:")
df.info()
print("\nDescriptive Statistics:")
print(df.describe().T)
        """, language="python")


def view_settings():
    """Settings View: Application preferences and configuration."""
    st.markdown("## ⚙️ Application Settings")
    st.markdown("Customize default behaviors for data generation and visualization.")

    col1, col2 = st.columns(2)
    with col1:
        st.number_input("Default Random Seed", value=42, key="pref_default_seed")
        st.selectbox("Default Plot Theme", ["Streamlit Light", "Plotly Dark", "Minimal Clean"], key="pref_plot_theme")
        st.number_input("Float Decimal Precision", min_value=1, max_value=6, value=2, key="pref_precision")
    with col2:
        st.selectbox("Missing Value Representation in JSON", ["null", "empty string", "NaN string"], key="pref_json_null")
        st.checkbox("Enable Developer Verbose Logging", value=False, key="pref_dev_mode")
        st.checkbox("Auto-repair Out-of-Bounds Values by Default", value=True, key="pref_auto_repair")

    if st.button("Save Settings", type="primary"):
        st.success("✅ Preferences saved for the current session.")


# =============================================================================
# Main Dispatcher
# =============================================================================
current_nav = st.session_state.get("current_nav", "Generate Dataset")

if current_nav == "Generate Dataset":
    view_generate_dataset()
elif current_nav == "Saved Datasets":
    view_saved_datasets()
elif current_nav == "Templates":
    view_templates()
elif current_nav == "Documentation":
    view_documentation()
elif current_nav == "Learn & Examples":
    view_learn_and_examples()
elif current_nav == "Settings":
    view_settings()
