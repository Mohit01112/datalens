
"""
DataLens – AI-Powered EDA & Dashboard Generator

Streamlit UI & orchestration layer.

LLM:
    Groq
    Model: loaded from .env through config.py

Environment variables:
    GROQ_API_KEY
    GROQ_MODEL
"""

from __future__ import annotations

from config import GROQ_API_KEY, GROQ_MODEL

import hashlib
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st


# =============================================================================
# APP-LEVEL LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

logger = logging.getLogger(__name__)


# =============================================================================
# LOCAL IMPORTS
# =============================================================================

from utils.data_loader import build_data_profile, load_data
from utils.eda_engine import build_charts, compute_kpis, sanitize_plan
from utils.filters import render_filters
from utils.groq_client import GroqClient


# =============================================================================
# OUTPUT DIRECTORIES
# =============================================================================

CODE_DIR = Path("outputs/generated_code")

CODE_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# GROQ CLIENT
# =============================================================================

def _get_groq_client() -> GroqClient:
    """
    Create the Groq client using credentials loaded from .env.

    The API key and model are NOT hard-coded here.
    """

    if not GROQ_API_KEY:
        st.error(
            "🔑 **GROQ_API_KEY not found.**\n\n"
            "Please add your Groq API key to the `.env` file."
        )
        st.stop()

    if not GROQ_MODEL:
        st.error(
            "⚠️ **GROQ_MODEL not found.**\n\n"
            "Please add the Groq model name to the `.env` file."
        )
        st.stop()

    return GroqClient(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
    )


# =============================================================================
# SESSION STATE
# =============================================================================

_DOWNSTREAM_KEYS = [
    "df",
    "profile",
    "filtered_df",
    "plan",
    "kpi_results",
    "chart_figures",
    "generated_code",
    "filename",
    "analysis_cache",
]


def _reset_downstream():
    """
    Clear all analysis results when a new file is uploaded.
    """

    for key in _DOWNSTREAM_KEYS:
        st.session_state.pop(key, None)


def _cache_key(profile: dict, user_request: str) -> str:
    """
    Generate a deterministic cache key using dataset profile
    and user's analysis request.
    """

    blob = (
        json.dumps(profile, sort_keys=True, default=str)
        + "|"
        + user_request.strip().lower()
    )

    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="DataLens – AI EDA & Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown(
    """
    <style>

    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* KPI card */
    .kpi-card {
        background: linear-gradient(
            135deg,
            #1e1e2f 0%,
            #2d2d44 100%
        );

        border: 1px solid rgba(255, 255, 255, 0.08);

        border-radius: 16px;

        padding: 1.4rem 1.2rem;

        text-align: center;

        box-shadow:
            0 4px 24px rgba(0, 0, 0, 0.25);

        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease;

        margin-bottom: 0.8rem;
    }

    .kpi-card:hover {
        transform: translateY(-4px);

        box-shadow:
            0 8px 32px rgba(99, 102, 241, 0.25);
    }

    .kpi-name {
        font-size: 0.78rem;

        color: #a0a0b8;

        text-transform: uppercase;

        letter-spacing: 0.08em;

        margin-bottom: 0.3rem;

        font-weight: 600;
    }

    .kpi-value {
        font-size: 1.8rem;

        font-weight: 800;

        background:
            linear-gradient(
                90deg,
                #818cf8,
                #6366f1
            );

        -webkit-background-clip: text;

        -webkit-text-fill-color: transparent;

        line-height: 1.2;
    }

    .kpi-meta {
        font-size: 0.7rem;

        color: #6b6b80;

        margin-top: 0.25rem;
    }

    .kpi-delta {
        font-size: 0.72rem;

        font-weight: 600;

        margin-top: 0.3rem;
    }

    .kpi-delta.positive {
        color: #34d399;
    }

    .kpi-delta.negative {
        color: #f87171;
    }

    .kpi-delta.neutral {
        color: #9ca3af;
    }

    /* Insights */
    .insight-item {
        background:
            rgba(99, 102, 241, 0.06);

        border-left:
            3px solid #6366f1;

        padding:
            0.7rem 1rem;

        margin-bottom:
            0.5rem;

        border-radius:
            0 8px 8px 0;

        font-size:
            0.92rem;

        color:
            #e0e0e8;
    }

    /* Section headers */
    .section-header {
        font-size:
            1.25rem;

        font-weight:
            700;

        color:
            #e0e0e8;

        margin:
            1.2rem 0 0.8rem 0;

        display:
            flex;

        align-items:
            center;

        gap:
            0.5rem;
    }

    /* Chart description */
    .chart-desc {
        font-size:
            0.8rem;

        color:
            #9ca3af;

        font-style:
            italic;

        margin-top:
            -0.4rem;

        margin-bottom:
            1rem;

        padding-left:
            0.3rem;
    }

    /* Column badges */
    .col-badge {
        display:
            inline-block;

        padding:
            2px 8px;

        border-radius:
            10px;

        font-size:
            0.7rem;

        font-weight:
            600;

        margin-right:
            4px;
    }

    .col-badge.numeric {
        background:
            #312e81;

        color:
            #a5b4fc;
    }

    .col-badge.categorical {
        background:
            #1e3a5f;

        color:
            #93c5fd;
    }

    .col-badge.date {
        background:
            #064e3b;

        color:
            #6ee7b7;
    }

    .col-badge.boolean {
        background:
            #4a1d96;

        color:
            #c4b5fd;
    }

    /* Progress */
    .progress-step {
        display:
            flex;

        align-items:
            center;

        gap:
            0.5rem;

        padding:
            0.4rem 0;

        font-size:
            0.88rem;
    }

    .progress-step .icon {
        font-size:
            1.1rem;
    }

    .progress-step.active {
        color:
            #818cf8;

        font-weight:
            600;
    }

    .progress-step.done {
        color:
            #34d399;
    }

    .progress-step.pending {
        color:
            #6b6b80;
    }

    /* Sidebar statistics */
    .sidebar-stat {
        display:
            flex;

        justify-content:
            space-between;

        padding:
            0.2rem 0;

        font-size:
            0.82rem;

        color:
            #a0a0b8;
    }

    .sidebar-stat .label {
        font-weight:
            500;
    }

    .sidebar-stat .value {
        font-weight:
            700;

        color:
            #e0e0e8;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:

    st.markdown("## 🔍 DataLens")

    st.markdown(
        "Upload a dataset and let AI discover the most useful insights."
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload your data",
        type=["csv", "xlsx", "xls"],
        help="Accepted formats: CSV, Excel (.xlsx / .xls)",
    )

    if uploaded_file is not None:

        # -------------------------------------------------------------
        # Reset state when a different file is uploaded
        # -------------------------------------------------------------

        if st.session_state.get("filename") != uploaded_file.name:

            _reset_downstream()

            st.session_state["filename"] = uploaded_file.name

        # -------------------------------------------------------------
        # Load dataset only once
        # -------------------------------------------------------------

        if "df" not in st.session_state:

            try:

                df = load_data(uploaded_file)

                st.session_state["df"] = df

                st.session_state["profile"] = build_data_profile(df)

            except Exception as exc:

                st.error(
                    f"⚠️ Could not load file: {exc}"
                )

                logger.error(
                    "File load error:\n%s",
                    traceback.format_exc(),
                )

                st.stop()

        df: pd.DataFrame = st.session_state["df"]

        st.success(
            f"✅ **{uploaded_file.name}** loaded"
        )

        # -------------------------------------------------------------
        # Dataset statistics
        # -------------------------------------------------------------

        profile = st.session_state["profile"]

        col_types = profile.get(
            "column_types",
            {}
        )

        n_numeric = sum(
            1
            for value in col_types.values()
            if value == "numeric"
        )

        n_cat = sum(
            1
            for value in col_types.values()
            if value == "categorical"
        )

        n_date = sum(
            1
            for value in col_types.values()
            if value == "date"
        )

        n_bool = sum(
            1
            for value in col_types.values()
            if value == "boolean"
        )

        avg_missing = (
            sum(
                profile.get(
                    "missing_pct",
                    {}
                ).values()
            )
            /
            max(
                len(
                    profile.get(
                        "missing_pct",
                        {}
                    )
                ),
                1,
            )
        )

        st.markdown(
            f"""
            <div class="sidebar-stat">
                <span class="label">Rows</span>
                <span class="value">{df.shape[0]:,}</span>
            </div>

            <div class="sidebar-stat">
                <span class="label">Columns</span>
                <span class="value">{df.shape[1]}</span>
            </div>

            <div class="sidebar-stat">
                <span class="label">📊 Numeric</span>
                <span class="value">{n_numeric}</span>
            </div>

            <div class="sidebar-stat">
                <span class="label">🏷️ Categorical</span>
                <span class="value">{n_cat}</span>
            </div>

            <div class="sidebar-stat">
                <span class="label">📅 Date</span>
                <span class="value">{n_date}</span>
            </div>

            <div class="sidebar-stat">
                <span class="label">🔘 Boolean</span>
                <span class="value">{n_bool}</span>
            </div>

            <div class="sidebar-stat">
                <span class="label">⚠️ Avg Missing</span>
                <span class="value">{avg_missing:.1f}%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # -------------------------------------------------------------
        # Data filters
        # -------------------------------------------------------------

        with st.expander(
            "🔎 Filter Data",
            expanded=False,
        ):

            filtered_df = render_filters(df)

            st.session_state["filtered_df"] = filtered_df

    st.divider()

    st.caption(
        f"Powered by Groq • {GROQ_MODEL}"
    )


# =============================================================================
# MAIN AREA
# =============================================================================

st.markdown(
    "# 🔍 DataLens"
)

st.markdown(
    "### AI-Powered EDA & Dashboard Generator"
)

st.caption(
    "Upload your dataset, describe what you want to discover, "
    "and let DataLens generate the analysis."
)


# =============================================================================
# CHECK DATASET
# =============================================================================

if "df" not in st.session_state:

    st.info(
        "👈 Upload a CSV or Excel file in the sidebar to get started."
    )

    st.stop()


df_raw = st.session_state["df"]

df = st.session_state.get(
    "filtered_df",
    df_raw,
)

profile = st.session_state["profile"]


# =============================================================================
# DATA PREVIEW
# =============================================================================

with st.expander(
    "🔍 Data Preview",
    expanded=False,
):

    col_types = profile.get(
        "column_types",
        {}
    )

    badge_map = {
        "numeric": ("📊", "numeric"),
        "categorical": ("🏷️", "categorical"),
        "date": ("📅", "date"),
        "boolean": ("🔘", "boolean"),
    }

    badges_html = " ".join(
        f"""
        <span class="col-badge {badge_map.get(ctype, ("", ""))[1]}">
            {badge_map.get(ctype, ("❓", ""))[0]} {col}
        </span>
        """
        for col, ctype in col_types.items()
    )

    st.markdown(
        badges_html,
        unsafe_allow_html=True,
    )

    st.dataframe(
        df.head(50),
        use_container_width=True,
    )

    # -------------------------------------------------------------
    # Missing values
    # -------------------------------------------------------------

    missing = profile.get(
        "missing_pct",
        {}
    )

    cols_with_missing = {
        key: value
        for key, value in missing.items()
        if value > 0
    }

    if cols_with_missing:

        st.caption(
            "⚠️ Columns with missing data: "
            +
            ", ".join(
                f"**{key}** ({value}%)"
                for key, value
                in cols_with_missing.items()
            )
        )

    st.caption(
        f"{df.shape[0]:,} rows × {df.shape[1]} columns"
    )


# =============================================================================
# USER REQUEST
# =============================================================================

st.markdown("---")

user_request = st.text_area(
    "What would you like to know about this data?",
    placeholder=(
        "e.g. 'Analyse the data', "
        "'Show me the most important KPIs', "
        "'What are the main trends?', "
        "'Compare sales by region' …"
    ),
    height=100,
)


analyze_clicked = st.button(
    "🚀 Analyze Dataset",
    type="primary",
    use_container_width=True,
)


# =============================================================================
# ANALYSIS PIPELINE
# =============================================================================

if analyze_clicked:

    if not user_request.strip():

        st.warning(
            "Please describe what you'd like to analyse."
        )

        st.stop()

    # -------------------------------------------------------------
    # Get Groq client
    # -------------------------------------------------------------

    groq = _get_groq_client()

    # -------------------------------------------------------------
    # Cache
    # -------------------------------------------------------------

    cache = st.session_state.get(
        "analysis_cache",
        {}
    )

    ck = _cache_key(
        profile,
        user_request,
    )

    # -------------------------------------------------------------
    # Load cached result
    # -------------------------------------------------------------

    if ck in cache:

        cached = cache[ck]

        st.session_state["plan"] = cached["plan"]

        st.session_state["kpi_results"] = (
            cached["kpi_results"]
        )

        st.session_state["chart_figures"] = (
            cached["chart_figures"]
        )

        st.session_state.pop(
            "generated_code",
            None,
        )

        st.toast(
            "⚡ Loaded from cache!",
            icon="⚡",
        )

    # -------------------------------------------------------------
    # New analysis
    # -------------------------------------------------------------

    else:

        for key in [
            "plan",
            "kpi_results",
            "chart_figures",
            "generated_code",
        ]:

            st.session_state.pop(
                key,
                None,
            )

        # ---------------------------------------------------------
        # Progress
        # ---------------------------------------------------------

        progress_container = st.container()

        progress_bar = st.progress(0)

        # ---------------------------------------------------------
        # Step 1 — AI Analysis Plan
        # ---------------------------------------------------------

        with progress_container:

            st.markdown(
                """
                <div class="progress-step active">
                    <span class="icon">🤖</span>
                    Step 1/3 — AI is designing the analysis plan…
                </div>
                """,
                unsafe_allow_html=True,
            )

        progress_bar.progress(10)

        try:

            # Build profile from filtered dataframe
            filtered_profile = build_data_profile(
                df
            )

            raw_plan = groq.get_analysis_plan(
                filtered_profile,
                user_request,
            )

        except json.JSONDecodeError:

            st.error(
                "⚠️ The AI returned malformed JSON. "
                "Please click **Analyze Dataset** again."
            )

            logger.error(
                "JSONDecodeError:\n%s",
                traceback.format_exc(),
            )

            st.stop()

        except Exception as exc:

            st.error(
                f"⚠️ Groq API error: {exc}"
            )

            logger.error(
                "Groq API error:\n%s",
                traceback.format_exc(),
            )

            st.stop()

        progress_bar.progress(40)

        # ---------------------------------------------------------
        # Step 2 — Validate AI plan
        # ---------------------------------------------------------

        plan = sanitize_plan(
            raw_plan,
            df,
        )

        st.session_state["plan"] = plan

        if not plan["kpis"] and not plan["charts"]:

            st.warning(
                "The AI analysis plan had no valid KPIs "
                "or charts after validation. "
                "Try rephrasing your request."
            )

            st.stop()

        # ---------------------------------------------------------
        # Step 2/3 — Compute KPIs
        # ---------------------------------------------------------

        with progress_container:

            st.markdown(
                """
                <div class="progress-step active">
                    <span class="icon">📐</span>
                    Step 2/3 — Computing KPIs from real data…
                </div>
                """,
                unsafe_allow_html=True,
            )

        progress_bar.progress(55)

        kpi_results = compute_kpis(
            plan["kpis"],
            df,
        )

        st.session_state[
            "kpi_results"
        ] = kpi_results

        # ---------------------------------------------------------
        # Step 3/3 — Build charts
        # ---------------------------------------------------------

        with progress_container:

            st.markdown(
                """
                <div class="progress-step active">
                    <span class="icon">📊</span>
                    Step 3/3 — Building interactive charts…
                </div>
                """,
                unsafe_allow_html=True,
            )

        progress_bar.progress(75)

        chart_figures = build_charts(
            plan["charts"],
            df,
        )

        st.session_state[
            "chart_figures"
        ] = chart_figures

        progress_bar.progress(100)

        # ---------------------------------------------------------
        # Save to cache
        # ---------------------------------------------------------

        if "analysis_cache" not in st.session_state:

            st.session_state[
                "analysis_cache"
            ] = {}

        st.session_state[
            "analysis_cache"
        ][ck] = {

            "plan": plan,

            "kpi_results": kpi_results,

            "chart_figures": chart_figures,
        }

        st.toast(
            "✅ Analysis complete!",
            icon="🎉",
        )


# =============================================================================
# RESULTS
# =============================================================================

if "kpi_results" in st.session_state:

    kpi_results = st.session_state[
        "kpi_results"
    ]

    plan = st.session_state[
        "plan"
    ]

    chart_figures = st.session_state.get(
        "chart_figures",
        [],
    )

    # -------------------------------------------------------------------------
    # Tabs
    # -------------------------------------------------------------------------

    tab_kpis, tab_charts, tab_insights, tab_code = st.tabs(
        [
            "📌 KPIs",
            "📊 Charts",
            "💡 Insights",
            "🧑‍💻 Code",
        ]
    )


    # =========================================================================
    # TAB 1 — KPIs
    # =========================================================================

    with tab_kpis:

        st.markdown(
            '<div class="section-header">'
            '📌 Key Performance Indicators'
            '</div>',
            unsafe_allow_html=True,
        )

        if kpi_results:

            cols = st.columns(
                min(
                    len(kpi_results),
                    4,
                )
            )

            for idx, kpi in enumerate(
                kpi_results
            ):

                col = cols[
                    idx % len(cols)
                ]

                with col:

                    delta = kpi.get(
                        "delta"
                    )

                    delta_label = kpi.get(
                        "delta_label",
                        "",
                    )

                    if delta is not None:

                        if delta > 0:

                            delta_html = (
                                f'<div class="kpi-delta positive">'
                                f'▲ {delta:+.1f}% '
                                f'{delta_label}'
                                f'</div>'
                            )

                        elif delta < 0:

                            delta_html = (
                                f'<div class="kpi-delta negative">'
                                f'▼ {delta:+.1f}% '
                                f'{delta_label}'
                                f'</div>'
                            )

                        else:

                            delta_html = (
                                f'<div class="kpi-delta neutral">'
                                f'● 0% '
                                f'{delta_label}'
                                f'</div>'
                            )

                    else:

                        delta_html = ""

                    st.html(
                        f"""
                        <div class="kpi-card">

                            <div class="kpi-name">
                                {kpi['name']}
                            </div>

                            <div class="kpi-value">
                                {kpi['formatted']}
                            </div>

                            <div class="kpi-meta">
                                {kpi['agg'].upper()}
                                ({kpi['column']})
                            </div>

                            {delta_html}

                        </div>
                        """
                    )

        else:

            st.info(
                "No KPIs were generated."
            )


    # =========================================================================
    # TAB 2 — CHARTS
    # =========================================================================

    with tab_charts:

        if chart_figures:

            st.markdown(
                '<div class="section-header">'
                '📊 Interactive Charts'
                '</div>',
                unsafe_allow_html=True,
            )

            for i in range(
                0,
                len(chart_figures),
                2,
            ):

                chart_cols = st.columns(2)

                for j, c in enumerate(
                    chart_cols
                ):

                    if i + j < len(
                        chart_figures
                    ):

                        fig, desc = (
                            chart_figures[
                                i + j
                            ]
                        )

                        with c:

                            st.plotly_chart(
                                fig,
                                use_container_width=True,
                                key=f"chart_{i+j}",
                            )

                            if desc:

                                st.markdown(
                                    f"""
                                    <div class="chart-desc">
                                        💬 {desc}
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

        else:

            st.info(
                "No charts were generated."
            )


    # =========================================================================
    # TAB 3 — AI INSIGHTS
    # =========================================================================

    with tab_insights:

        insights = plan.get(
            "insights",
            [],
        )

        if insights:

            st.markdown(
                '<div class="section-header">'
                '💡 AI-Generated Insights'
                '</div>',
                unsafe_allow_html=True,
            )

            st.caption(
                "These observations are AI-generated "
                "commentary grounded in the dataset profile."
            )

            for insight in insights:

                st.markdown(
                    f"""
                    <div class="insight-item">
                        {insight}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        else:

            st.info(
                "No insights were generated."
            )


    # =========================================================================
    # TAB 4 — CODE GENERATION
    # =========================================================================

    with tab_code:

        st.markdown(
            '<div class="section-header">'
            '🧑‍💻 EDA Code Generator'
            '</div>',
            unsafe_allow_html=True,
        )

        st.caption(
            "Generate a standalone Python script "
            "that reproduces this analysis."
        )

        generate_code_clicked = st.button(
            "⚡ Generate EDA Code",
            use_container_width=True,
            help=(
                "Generate a standalone Python script "
                "(pandas + matplotlib) for this analysis."
            ),
            key="btn_generate_code",
        )

        if generate_code_clicked:

            groq = _get_groq_client()

            with st.spinner(
                "🧑‍💻 Generating EDA script…"
            ):

                try:

                    code = groq.generate_eda_code(
                        plan,
                        profile,
                        st.session_state.get(
                            "filename",
                            "data.csv",
                        ),
                    )

                    st.session_state[
                        "generated_code"
                    ] = code

                    # Save generated script
                    ts = datetime.now().strftime(
                        "%Y%m%d_%H%M%S"
                    )

                    save_path = (
                        CODE_DIR
                        /
                        f"eda_script_{ts}.py"
                    )

                    save_path.write_text(
                        code,
                        encoding="utf-8",
                    )

                    logger.info(
                        "EDA script saved to %s",
                        save_path,
                    )

                except Exception as exc:

                    st.error(
                        f"⚠️ Code generation failed: {exc}"
                    )

                    logger.error(
                        "Code generation error:\n%s",
                        traceback.format_exc(),
                    )

        # -------------------------------------------------------------
        # Display generated code
        # -------------------------------------------------------------

        if "generated_code" in st.session_state:

            code = st.session_state[
                "generated_code"
            ]

            st.code(
                code,
                language="python",
            )

            st.download_button(
                "⬇️ Download Python Script",
                data=code,
                file_name="eda_analysis.py",
                mime="text/x-python",
                key="dl_code",
            )


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")

st.caption(
    f"🔍 DataLens • Powered by Groq • {GROQ_MODEL}"
)
