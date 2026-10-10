"""Dataset Preview Component.

Renders the Dataset Preview card with row/column count badges,
interactive tabular view, pagination controls, and empty-state placeholders.
"""

from __future__ import annotations
from typing import Optional
import pandas as pd
import streamlit as st


def render_preview_panel(df: Optional[pd.DataFrame]) -> None:
    """Render the Dataset Preview card matching the design mockup."""
    if df is None or df.empty:
        # Empty placeholder state
        st.markdown("""
            <div class="preview-card-container">
                <div class="card-header-row">
                    <div class="card-title">
                        <span class="icon">▦</span> Dataset Preview
                    </div>
                    <div class="badges-row">
                        <span class="badge badge-blue">0 rows</span>
                        <span class="badge badge-blue">0 columns</span>
                    </div>
                </div>
                <div class="empty-state-box">
                    <p style="color: #64748b; font-size: 0.95rem; margin: 2rem 0; text-align: center;">
                        👈 Configure your parameters on the left and click <b>Generate Dataset</b> to preview synthetic records here.
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        return

    n_rows, n_cols = df.shape

    # Header with title and pill badges
    st.markdown(f"""
        <div class="card-header-row">
            <div class="card-title">
                <span class="icon">▦</span> Dataset Preview
            </div>
            <div class="badges-row">
                <span class="badge badge-blue">{n_rows:,} rows</span>
                <span class="badge badge-blue">{n_cols} columns</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Preview controls (page size and search/filter)
    col1, col2 = st.columns([2, 3])
    with col1:
        page_size = st.selectbox(
            "Rows per page",
            options=[10, 25, 50, 100],
            index=0,
            key="preview_page_size",
            label_visibility="collapsed",
        )
    with col2:
        search_term = st.text_input(
            "Filter records",
            placeholder="🔍 Filter preview by keyword...",
            key="preview_filter_search",
            label_visibility="collapsed",
        )

    # Filter df if search term provided
    display_df = df
    if search_term.strip():
        mask = display_df.astype(str).apply(lambda row: row.str.contains(search_term, case=False).any(), axis=1)
        display_df = display_df[mask]

    # Format numeric columns for clean tabular preview
    preview_slice = display_df.head(page_size)

    # Render Streamlit Dataframe with styled height and full container width
    st.dataframe(
        preview_slice,
        use_container_width=True,
        hide_index=True,
        height=min(400, 38 + len(preview_slice) * 35),
    )

    st.caption(f"Showing {len(preview_slice)} of {n_rows:,} total records. All {n_rows:,} rows are available for download.")
