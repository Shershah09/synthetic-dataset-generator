"""Quality dashboard and analytics component.

Renders:
1. Class Distribution donut chart (or regression/clustering target chart)
2. Correlation Heatmap
3. Feature Distributions histogram with feature selector
4. Dataset Statistics table
5. Download Dataset panel with CSV, Excel, and JSON export buttons
6. Comprehensive Data Quality & Validation Audit Report
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from synthetic_dataset_generator.core.exporter import ExportService
from synthetic_dataset_generator.core.schema_builder import DatasetSchema
from synthetic_dataset_generator.core.validator import CheckStatus, QualityReport


def render_analytics_and_downloads(
    df: Optional[pd.DataFrame],
    schema: Optional[DatasetSchema],
    report: Optional[QualityReport],
) -> None:
    """Render the analytics charts, statistics table, and download panel."""
    if df is None or df.empty or schema is None:
        return

    # =========================================================================
    # ROW 1: 3 Analytics Cards (Class Distribution, Heatmap, Feature Distribution)
    # =========================================================================
    col_chart1, col_chart2, col_chart3 = st.columns([1, 1, 1], gap="small")

    # -------------------------------------------------------------------------
    # Chart 1: Target / Class Distribution
    # -------------------------------------------------------------------------
    with col_chart1:
        st.markdown("""
            <div class="card-header-row" style="margin-bottom: 0.2rem;">
                <div class="card-title" style="font-size: 0.95rem;">
                    <span class="icon">📊</span> Class Distribution
                </div>
            </div>
        """, unsafe_allow_html=True)

        target_col = schema.target_column
        if schema.ml_task == "Classification" and target_col and target_col in df.columns:
            target_series = df[target_col].dropna()
            val_counts = target_series.value_counts().reset_index()
            val_counts.columns = ["Class", "Count"]

            # Label mapping for clean display
            class_labels = {
                0: "No Disease (0)",
                1: "Disease (1)",
                "0": "No Disease (0)",
                "1": "Disease (1)",
            }
            val_counts["Label"] = val_counts["Class"].apply(lambda c: class_labels.get(c, f"Class {c}"))

            total_samples = len(target_series)
            fig_donut = go.Figure(
                data=[
                    go.Pie(
                        labels=val_counts["Label"],
                        values=val_counts["Count"],
                        hole=0.65,
                        marker=dict(colors=["#3b82f6", "#f43f5e", "#10b981", "#f59e0b"]),
                        textinfo="percent",
                        hoverinfo="label+value+percent",
                        showlegend=True,
                    )
                ]
            )
            # Add center text e.g. "1000 Samples"
            fig_donut.add_annotation(
                text=f"<b>{total_samples:,}</b><br><span style='font-size: 11px; color: #64748b;'>Samples</span>",
                x=0.5,
                y=0.5,
                font=dict(size=14, color="#1e293b"),
                showarrow=False,
            )
            fig_donut.update_layout(
                margin=dict(l=5, r=5, t=10, b=10),
                height=230,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.02,
                    font=dict(size=10),
                ),
            )
            st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})

        elif schema.ml_task == "Regression" and target_col and target_col in df.columns:
            # Continuous target distribution
            fig_reg = px.histogram(
                df,
                x=target_col,
                nbins=20,
                marginal="box",
                color_discrete_sequence=["#3b82f6"],
            )
            fig_reg.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                height=230,
                xaxis_title=target_col,
                yaxis_title="Count",
            )
            st.plotly_chart(fig_reg, use_container_width=True, config={"displayModeBar": False})

        elif schema.ml_task == "Clustering":
            # Show clustering breakdown
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and "ID" not in c.upper()]
            if len(num_cols) >= 2:
                fig_scatter = px.scatter(
                    df,
                    x=num_cols[0],
                    y=num_cols[1],
                    opacity=0.7,
                    color_discrete_sequence=["#3b82f6"],
                )
                fig_scatter.update_layout(
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=230,
                )
                st.plotly_chart(fig_scatter, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("Insufficient numeric dimensions for cluster plot.")
        else:
            st.info("No target variable selected.")

    # -------------------------------------------------------------------------
    # Chart 2: Correlation Heatmap
    # -------------------------------------------------------------------------
    with col_chart2:
        st.markdown("""
            <div class="card-header-row" style="margin-bottom: 0.2rem;">
                <div class="card-title" style="font-size: 0.95rem;">
                    <span class="icon">⊞</span> Correlation Heatmap
                </div>
            </div>
        """, unsafe_allow_html=True)

        numeric_df = df.select_dtypes(include=[np.number]).copy()
        # Drop ID columns from heatmap
        for c in numeric_df.columns:
            if "ID" in c.upper():
                numeric_df.drop(columns=[c], inplace=True)

        if len(numeric_df.columns) >= 2:
            # Pick top 5 numeric columns for clean layout matching screenshot
            cols_for_corr = numeric_df.columns[:5]
            corr_matrix = numeric_df[cols_for_corr].corr().round(2)

            fig_heat = px.imshow(
                corr_matrix,
                text_auto=False,
                aspect="auto",
                color_continuous_scale="RdBu_r",
                zmin=-1.0,
                zmax=1.0,
            )
            fig_heat.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                height=230,
                coloraxis_colorbar=dict(
                    thickness=10,
                    len=0.9,
                    tickfont=dict(size=9),
                ),
                xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
                yaxis=dict(tickfont=dict(size=9)),
            )
            st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("At least 2 numeric features needed for correlation heatmap.")

    # -------------------------------------------------------------------------
    # Chart 3: Feature Distributions
    # -------------------------------------------------------------------------
    with col_chart3:
        # Header with title and feature dropdown
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and "ID" not in c.upper()]
        selected_feature = None

        if num_cols:
            feat_col1, feat_col2 = st.columns([1.2, 1])
            with feat_col1:
                st.markdown("""
                    <div class="card-header-row" style="margin-bottom: 0;">
                        <div class="card-title" style="font-size: 0.95rem;">
                            <span class="icon">📊</span> Feature Distributions
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            with feat_col2:
                selected_feature = st.selectbox(
                    "Feature",
                    options=num_cols,
                    index=0,
                    key="select_dist_feature",
                    label_visibility="collapsed",
                )

            # Histogram matching the UI reference
            fig_hist = px.histogram(
                df,
                x=selected_feature,
                nbins=22,
                color_discrete_sequence=["#3b82f6"],
            )
            fig_hist.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                height=230,
                xaxis_title=selected_feature,
                yaxis_title="Count",
                bargap=0.1,
                xaxis=dict(tickfont=dict(size=9)),
                yaxis=dict(tickfont=dict(size=9)),
            )
            st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No numeric features available for distribution histogram.")

    # =========================================================================
    # ROW 2: Dataset Statistics & Download Panel
    # =========================================================================
    col_stats, col_downloads = st.columns([1.75, 1.0], gap="medium")

    # -------------------------------------------------------------------------
    # Panel A: Dataset Statistics Table
    # -------------------------------------------------------------------------
    with col_stats:
        st.markdown("""
            <div class="card-header-row">
                <div class="card-title">
                    <span class="icon">📄</span> Dataset Statistics
                </div>
            </div>
        """, unsafe_allow_html=True)

        if report and report.column_stats:
            stats_list = []
            for col_name, stats in report.column_stats.items():
                stats_list.append({
                    "Column": col_name,
                    "Type": stats["type"],
                    "Non-Null": stats["non_null"],
                    "Missing %": stats["missing_pct"],
                    "Mean": stats["mean"],
                    "Std": stats["std"],
                    "Min": stats["min"],
                    "Max": stats["max"],
                })
            stats_df = pd.DataFrame(stats_list)
        else:
            # Fallback computation
            stats_list = []
            for col in df.columns:
                is_num = pd.api.types.is_numeric_dtype(df[col])
                non_null = int(df[col].notna().sum())
                miss_pct = round(float((df[col].isna().sum() / len(df)) * 100), 1)
                stats_list.append({
                    "Column": col,
                    "Type": "Numeric" if is_num else "Categorical",
                    "Non-Null": non_null,
                    "Missing %": miss_pct,
                    "Mean": round(float(df[col].mean()), 1) if is_num else "-",
                    "Std": round(float(df[col].std()), 1) if is_num else "-",
                    "Min": round(float(df[col].min()), 1) if is_num else "-",
                    "Max": round(float(df[col].max()), 1) if is_num else "-",
                })
            stats_df = pd.DataFrame(stats_list)

        st.dataframe(
            stats_df,
            use_container_width=True,
            hide_index=True,
            height=240,
        )

    # -------------------------------------------------------------------------
    # Panel B: Download Dataset Panel
    # -------------------------------------------------------------------------
    with col_downloads:
        st.markdown("""
            <div class="card-header-row">
                <div class="card-title">
                    <span class="icon">📥</span> Download Dataset
                </div>
            </div>
        """, unsafe_allow_html=True)

        # In-memory export generation for CSV, Excel, and JSON
        domain = schema.domain
        task = schema.ml_task

        csv_bytes, csv_mime, csv_name = ExportService.to_csv(df, domain, task)
        excel_bytes, excel_mime, excel_name = ExportService.to_excel(df, domain, task)
        json_bytes, json_mime, json_name = ExportService.to_json(df, domain, task)

        # CSV Download Button (Green)
        st.download_button(
            label="📄 Download as CSV",
            data=csv_bytes,
            file_name=csv_name,
            mime=csv_mime,
            use_container_width=True,
            key="dl_btn_csv",
        )

        # Excel Download Button (Blue)
        st.download_button(
            label="📊 Download as Excel",
            data=excel_bytes,
            file_name=excel_name,
            mime=excel_mime,
            use_container_width=True,
            key="dl_btn_excel",
        )

        # JSON Download Button (Purple)
        st.download_button(
            label="{ } Download as JSON",
            data=json_bytes,
            file_name=json_name,
            mime=json_mime,
            use_container_width=True,
            key="dl_btn_json",
        )

        st.caption("🔒 All 100% generated in-memory. Ready for ML/DL pipelines.")

    # =========================================================================
    # ROW 3: Quality Audit & Validation Drawer
    # =========================================================================
    if report:
        with st.expander(f"🛡 Data Quality & Validation Report ({report.overall_score}/100)", expanded=False):
            q_col1, q_col2, q_col3, q_col4 = st.columns(4)
            q_col1.metric("Overall Quality Score", f"{report.overall_score}/100")
            q_col2.metric("Checks Passed", report.passed_count)
            q_col3.metric("Warnings Flagged", report.warning_count)
            q_col4.metric("Failures Detected", report.failed_count)

            st.markdown("#### Quality Audit Checkpoints")
            for chk in report.checks:
                if chk.status == CheckStatus.PASSED:
                    status_icon = "🟢 **PASSED**"
                elif chk.status == CheckStatus.WARNING:
                    status_icon = "🟡 **WARNING**"
                else:
                    status_icon = "🔴 **FAILED**"

                repair_note = " *(Auto-repaired)*" if chk.repaired else ""
                st.markdown(f"- {status_icon} **{chk.check_name}** ({chk.category}): {chk.details}{repair_note}")

            st.info(f"ℹ️ **Evaluation Note**: {report.statistical_realism_note}")
