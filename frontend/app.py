import os
import json
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Local module imports
from frontend.components.valuation_chart import create_valuation_scatter_matrix
from frontend.components.radars import create_radar_comparison
from frontend.components.pitch_maps import create_tracking_pitch_figure, create_tactical_formation_board
from core_ml.data_loader import load_raw_dataset
from core_ml.squad_optimizer import SquadOptimizer
from core_ml.explainability import MarketValueExplainer
from api.services.vector_search import vector_search_service
from api.services.scout_agent import ScoutAgentService
from config.settings import settings

logger = logging.getLogger("football_moneyball_app")

# Page Configuration
st.set_page_config(
    page_title="Football Moneyball 2.0 | AI Scouting Decision Intelligence",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Scouting Dark Theme Styling
st.markdown(
    """
    <style>
    .main {
        background-color: #0B1120;
    }
    .stMetric {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 12px 18px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .stMetric label {
        color: #94A3B8 !important;
        font-weight: 500;
        font-size: 0.85rem;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #F8FAFC !important;
        font-weight: 700;
    }
    .executive-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .badge-gem {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10B981;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .badge-star {
        background-color: rgba(56, 189, 248, 0.2);
        color: #38BDF8;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
        border: 1px solid rgba(56, 189, 248, 0.4);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_backend_services():
    """Initializes and caches core backend ML and Search services."""
    explainer = MarketValueExplainer()
    scout_agent = ScoutAgentService()
    players = load_raw_dataset()
    optimizer = SquadOptimizer(players)
    return explainer, scout_agent, optimizer


@st.cache_data
def load_dataset() -> List[Dict[str, Any]]:
    """Loads player dataset enriched with valuations and tactical embeddings."""
    return load_raw_dataset()


# Initialize services
explainer, scout_agent, squad_optimizer = load_backend_services()
all_players = load_dataset()

# Top Header Bar
col_logo, col_stat = st.columns([4, 1])
with col_logo:
    st.title("⚽ Football Moneyball 2.0 & Scouting Decision Intelligence")
    st.caption("AI-Powered Player Valuation Arbitrage, HNSW Tactical Clones & Gemini 3.8 Flash Dossiers")
with col_stat:
    api_status = "Online (Local Mock Mode)" if not settings.GEMINI_API_KEY else "Online (Gemini 3.8 Live)"
    st.metric(label="Scout Engine Status", value=api_status, delta="Postgres / Vector 64-d")

# Navigation Tabs
tabs = st.tabs([
    "📊 Valuation Arbitrage Matrix",
    "🧬 Tactical Clone Finder",
    "🤖 AI Transfer Audit Dossier",
    "📹 CV Scout Video-to-Data",
    "🏆 Moneyball Squad Optimizer",
])

# ==============================================================================
# TAB 1: VALUATION ARBITRAGE MATRIX
# ==============================================================================
with tabs[0]:
    st.subheader("Market Inefficiency & Valuation Arbitrage (LightGBM Core)")
    st.write(
        "Compare actual Transfermarkt valuations vs **Moneyball 2.0 Fair Value** predicted purely from "
        "age, minutes played, and comprehensive FBref/StatsBomb p90 performance metrics."
    )

    # Filter row
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        leagues = sorted(list(set(p.get("league", "") for p in all_players)))
        selected_leagues = st.multiselect("Leagues", options=leagues, default=leagues[:5])
    with f_col2:
        positions = ["ALL", "CB", "FB", "CM", "CAM", "W", "ST"]
        selected_pos = st.selectbox("Position", options=positions, index=0)
    with f_col3:
        min_perf = st.slider("Min Performance Score", min_value=40.0, max_value=95.0, value=55.0, step=5.0)
    with f_col4:
        search_query = st.text_input("Search Player Name", placeholder="e.g. Rodri, Saka, Hjulmand...")

    # Filter logic
    filtered_players = []
    for p in all_players:
        if selected_leagues and p.get("league") not in selected_leagues:
            continue
        if selected_pos != "ALL" and p.get("primary_position") != selected_pos:
            continue
        if p.get("performance_score", 0) < min_perf:
            continue
        if search_query and search_query.lower() not in p.get("name", "").lower():
            continue
        filtered_players.append(p)

    # High-level Metrics Row
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Filtered Candidates", len(filtered_players), f"Total: {len(all_players)}")
    with m_col2:
        gems = [p for p in filtered_players if p.get("valuation", {}).get("undervaluation_delta_eur", 0) > 5_000_000]
        st.metric("Undervalued Gems Identified", len(gems), f"{len(gems)/max(1, len(filtered_players))*100:.1f}%")
    with m_col3:
        avg_delta = np.mean([p.get("valuation", {}).get("undervaluation_delta_eur", 0) for p in filtered_players]) if filtered_players else 0
        st.metric("Avg Arbitrage Delta", f"€{avg_delta / 1_000_000:+.2f}M")
    with m_col4:
        best_gem = max(filtered_players, key=lambda x: x.get("valuation", {}).get("undervaluation_delta_eur", 0)) if filtered_players else None
        gem_name = f"{best_gem['name']} (+€{best_gem['valuation']['undervaluation_delta_eur']/1e6:.1f}M)" if best_gem else "N/A"
        st.metric("Top Arbitrage Target", gem_name)

    # Valuation Scatter Plot
    fig_val = create_valuation_scatter_matrix(filtered_players)
    st.plotly_chart(fig_val, use_container_width=True)

    # Top Undervalued Gems Table
    st.markdown("### 💎 Top Identified Moneyball 2.0 Targets")
    top_gems = sorted(
        filtered_players,
        key=lambda x: x.get("valuation", {}).get("undervaluation_delta_eur", 0),
        reverse=True
    )[:15]

    table_data = []
    for p in top_gems:
        v = p.get("valuation", {})
        table_data.append({
            "Player": p["name"],
            "Club": p["club"],
            "League": p["league"],
            "Pos": p["primary_position"],
            "Age": p["age"],
            "Actual Price (€M)": round(p["market_value_eur"] / 1e6, 2),
            "Moneyball Value (€M)": round(v.get("predicted_value_eur", 0) / 1e6, 2),
            "Arbitrage Delta (€M)": round(v.get("undervaluation_delta_eur", 0) / 1e6, 2),
            "Perf Score": p.get("performance_score", 0),
        })

    if table_data:
        df_gems = pd.DataFrame(table_data)
        st.dataframe(df_gems, use_container_width=True, hide_index=True)


# ==============================================================================
# TAB 2: TACTICAL CLONE FINDER
# ==============================================================================
with tabs[1]:
    st.subheader("🧬 Tactical Clone Finder & DNA Radar Matching")
    st.write(
        "Find high-performing tactical analogs of elite stars from lower-cost donor leagues "
        "using 64-dimensional HNSW cosine embeddings."
    )

    # Target player selection
    col_t1, col_t2 = st.columns([2, 1])
    player_names = sorted(list(set(p["name"] for p in all_players)))
    default_idx = player_names.index("Rodri") if "Rodri" in player_names else 0

    with col_t1:
        target_name = st.selectbox("Select Target Star to Clone", options=player_names, index=default_idx)
    with col_t2:
        budget_limit = st.number_input("Max Clone Market Value (€M)", min_value=5.0, max_value=200.0, value=60.0, step=5.0)

    target_p = next((p for p in all_players if p["name"] == target_name), None)

    if target_p:
        # Search for analogs using vector service
        max_cost_eur = budget_limit * 1_000_000.0
        analogs_resp = vector_search_service.find_tactical_analogs(
            target_player_id=target_p["id"],
            top_k=5,
            same_position=True,
            max_market_value=max_cost_eur,
        )
        analogs = [a.model_dump() for a in analogs_resp.analogs]

        st.markdown(
            f"""
            <div class="executive-card">
                <span class="badge-star">Target Benchmark Star</span>
                <h3 style="margin-top: 8px; margin-bottom: 4px; color: #38BDF8;">{target_p['name']} ({target_p['club']})</h3>
                <p style="color: #94A3B8; margin-bottom: 0;">
                    <b>Position:</b> {target_p['primary_position']} | 
                    <b>Age:</b> {target_p['age']} | 
                    <b>Market Price:</b> €{target_p['market_value_eur']/1e6:.1f}M | 
                    <b>Perf Score:</b> {target_p.get('performance_score', 0)}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not analogs:
            st.warning("No tactical clones match the criteria within this budget limit. Try increasing the max budget.")
        else:
            col_list, col_radar = st.columns([1, 1])

            with col_list:
                st.markdown("### 🏆 Top Tactical Clones Found")
                clone_options = [f"{a['name']} ({a['club']}) - Sim: {a['similarity_score_pct']:.1f}%" for a in analogs]
                chosen_idx = st.radio("Select Candidate to Compare:", range(len(clone_options)), format_func=lambda i: clone_options[i])
                selected_clone_summary = analogs[chosen_idx]
                selected_clone = vector_search_service.get_player(selected_clone_summary["id"])

                # Financial arbitrage summary
                star_price = target_p["market_value_eur"]
                clone_price = selected_clone["market_value_eur"]
                savings = star_price - clone_price
                pct_saved = (savings / star_price) * 100.0 if star_price > 0 else 0

                st.markdown(
                    f"""
                    <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 14px; margin-top: 15px;">
                        <h4 style="color: #10B981; margin: 0 0 6px 0;">💰 Transfer Arbitrage Summary</h4>
                        <p style="margin: 0; color: #E2E8F0; font-size: 0.95rem;">
                            Acquiring <b>{selected_clone['name']}</b> achieves <b>{selected_clone_summary['similarity_score_pct']:.1f}%</b> tactical role duplication 
                            while saving <b style="color: #10B981;">€{savings/1e6:.1f}M ({pct_saved:.1f}%)</b> on transfer fee capital outlay!
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col_radar:
                radar_fig = create_radar_comparison(
                    target_player=target_p,
                    clone_player=selected_clone,
                    similarity_score=selected_clone_summary["similarity_score_pct"] / 100.0,
                )
                st.plotly_chart(radar_fig, use_container_width=True)


# ==============================================================================
# TAB 3: AI TRANSFER AUDIT DOSSIER (GEMINI 3.8 FLASH)
# ==============================================================================
with tabs[2]:
    st.subheader("🤖 AI Transfer Audit Dossier (Gemini 3.8 Flash & SHAP XAI)")
    st.write(
        "Generate rigorous, board-level transfer scouting audit reports powered by Google Gemini 3.8 Flash "
        "and TreeSHAP feature attribution."
    )

    audit_c1, audit_c2, audit_c3 = st.columns([1, 1, 2])
    with audit_c1:
        star_for_audit = st.selectbox("Benchmark Target Star", options=player_names, index=default_idx, key="audit_star")
    with audit_c2:
        target_obj = next((p for p in all_players if p["name"] == star_for_audit), None)
        analogs_for_star = vector_search_service.find_tactical_analogs(
            target_player_id=target_obj["id"],
            top_k=5,
            max_market_value=60_000_000.0,
        )
        clone_names = [a.name for a in analogs_for_star.analogs] if analogs_for_star.analogs else [p["name"] for p in all_players[:5]]
        selected_clone_name = st.selectbox("Analog Candidate", options=clone_names, index=0, key="audit_clone")
    with audit_c3:
        custom_tactics = st.text_input("Custom Tactical Instructions (Optional)", placeholder="e.g. Focus on high press transition and ball retention under pressure")

    audit_target_p = next((p for p in all_players if p["name"] == star_for_audit), None)
    audit_clone_p = next((p for p in all_players if p["name"] == selected_clone_name), None)

    if st.button("🚀 Generate Executive Transfer Audit Dossier", type="primary"):
        if audit_target_p and audit_clone_p:
            with st.spinner("Analyzing tactical DNA, SHAP feature importance, and generating Gemini 3.8 audit..."):
                sim_res = vector_search_service.find_tactical_analogs(
                    target_player_id=audit_target_p["id"],
                    top_k=10,
                )
                matched_analog = next((a for a in sim_res.analogs if a.id == audit_clone_p["id"]), None)
                sim_pct = matched_analog.similarity_score_pct if matched_analog else 88.0

                shap_details = explainer.explain_player(audit_clone_p)

                dossier = scout_agent.generate_dossier(
                    target_player=audit_target_p,
                    analog_player=audit_clone_p,
                    similarity_score_pct=sim_pct,
                    shap_drivers=shap_details,
                    custom_instructions=custom_tactics or None,
                )

                st.session_state["latest_dossier"] = dossier
                st.session_state["audit_clone_p"] = audit_clone_p

    if "latest_dossier" in st.session_state:
        dossier = st.session_state["latest_dossier"]
        clone_p = st.session_state.get("audit_clone_p", audit_clone_p)

        # Executive Metrics Header
        st.markdown("---")
        d_c1, d_c2, d_c3, d_c4 = st.columns(4)
        with d_c1:
            st.metric("Executive Recommendation", dossier.verdict)
        with d_c2:
            st.metric("Tactical Similarity Score", f"{dossier.tactical_similarity_score:.1f}%")
        with d_c3:
            st.metric("Capital Arbitrage Delta", f"€{dossier.financial_arbitrage.expected_savings_eur/1e6:+.1f}M")
        with d_c4:
            st.metric("Confidence Level", f"{dossier.confidence_level_pct:.0f}%")

        # Executive Summary & Tactical Alignment
        e_col1, e_col2 = st.columns([1, 1])
        with e_col1:
            st.markdown(
                f"""
                <div class="executive-card">
                    <h4 style="color: #38BDF8; margin-top: 0;">📋 Executive Scouting Summary</h4>
                    <p style="color: #F8FAFC; line-height: 1.6;">{dossier.executive_summary}</p>
                    <h5 style="color: #10B981;">⚡ Key Tactical Strengths</h5>
                    <ul>
                        {"".join(f"<li style='color: #E2E8F0;'><b>{k.metric}</b> ({k.percentile:.0f}th percentile): {k.tactical_rationale}</li>" for k in dossier.key_strengths)}
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with e_col2:
            # SHAP Explainability waterfall / bar chart
            shap_data = explainer.explain_player(clone_p)
            top_pos = shap_data.get("top_positive_drivers", [])
            top_neg = shap_data.get("top_negative_drivers", [])

            all_drivers = top_pos + top_neg
            shap_features = [d.get("label", d.get("feature", "")) for d in all_drivers]
            shap_values = [d.get("impact_eur", 0) / 1_000_000.0 for d in all_drivers]
            bar_colors = ["#10B981" if v >= 0 else "#EF4444" for v in shap_values]

            fig_shap = go.Figure(
                go.Bar(
                    x=shap_values,
                    y=shap_features,
                    orientation="h",
                    marker=dict(color=bar_colors),
                )
            )
            fig_shap.update_layout(
                title="<b>SHAP Feature Attribution: Fair Value Drivers (€M)</b>",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.4)",
                font=dict(color="#F8FAFC"),
                height=320,
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis=dict(title="Euro Valuation Impact (€M)"),
            )
            st.plotly_chart(fig_shap, use_container_width=True)

        # Risk Assessment & Potential Resale Profit
        st.markdown("### ⚠️ Strategic Risk Matrix & Projected Resale Profit")
        r_c1, r_c2 = st.columns(2)
        with r_c1:
            for risk in dossier.risks_and_mitigations:
                st.warning(f"**{risk.category.upper()} ({risk.severity})**: {risk.details}\n\n*Mitigation:* {risk.mitigation}")
        with r_c2:
            st.info(f"**Projected Resale Profit:** €{dossier.financial_arbitrage.projected_resale_profit_eur/1e6:.1f}M within 2-3 seasons.\n\n**Tactical Fit Analysis:**\n{dossier.tactical_fit_analysis}")

        # Download Markdown Dossier
        md_text = f"""# Transfer Audit Dossier: {dossier.recommended_analog_name}
**Target Benchmark Star:** {dossier.target_player_name}
**Verdict:** {dossier.verdict}
**Tactical Similarity:** {dossier.tactical_similarity_score:.1f}%
**Expected Transfer Savings:** €{dossier.financial_arbitrage.expected_savings_eur/1e6:.1f}M
**Projected Resale Profit:** €{dossier.financial_arbitrage.projected_resale_profit_eur/1e6:.1f}M

## Executive Summary
{dossier.executive_summary}

## Key Tactical Strengths
""" + "\n".join(f"- **{k.metric}** ({k.percentile:.0f}th percentile): {k.tactical_rationale}" for k in dossier.key_strengths) + f"""

## Tactical Fit Analysis
{dossier.tactical_fit_analysis}

## Risk Matrix
""" + "\n".join(f"- **{r.category.upper()} [{r.severity}]:** {r.details} (Mitigation: {r.mitigation})" for r in dossier.risks_and_mitigations)

        st.download_button(
            "📥 Download Executive Dossier (Markdown)",
            data=md_text,
            file_name=f"transfer_audit_{dossier.recommended_analog_name.lower().replace(' ', '_')}.md",
            mime="text/markdown",
        )


# ==============================================================================
# TAB 4: CV SCOUT VIDEO-TO-DATA INTELLIGENCE
# ==============================================================================
with tabs[3]:
    st.subheader("📹 CV Scout Video-to-Data Tracking & Pitch Homography")
    st.write(
        "Ingests optical tracking frames from match broadcast video. "
        "YOLOv11 detects players and the match ball, ByteTrack maintains consistent IDs across occlusions, "
        "and a 4-point projective homography maps coordinates onto the standard **105m x 68m FIFA pitch**."
    )

    sample_tracking_path = os.path.join(settings.BASE_DIR, "cv_module", "samples", "scout_tracking_poc.json")
    if os.path.exists(sample_tracking_path):
        with open(sample_tracking_path, "r") as f:
            cv_tracking = json.load(f)

        frames = cv_tracking.get("frames", [])
        speeds = []
        all_players_in_frames = set()
        for fr in frames:
            for pl in fr.get("players", []):
                speeds.append(pl.get("speed_kmh", 0.0))
                all_players_in_frames.add(pl.get("track_id"))

        max_speed = max(speeds) if speeds else 0.0
        sprints = sum(1 for s in speeds if s >= 25.0)

        # CV Metrics Row
        cv_c1, cv_c2, cv_c3, cv_c4 = st.columns(4)
        with cv_c1:
            st.metric("Optical Frames Analyzed", len(frames), "30 FPS")
        with cv_c2:
            st.metric("Unique Track IDs", len(all_players_in_frames), "ByteTrack Continuous")
        with cv_c3:
            st.metric("Max Sprint Recorded", f"{max_speed:.1f} km/h")
        with cv_c4:
            st.metric("High Intensity Sprints", sprints, ">25 km/h")

        # 2D FIFA Pitch Map with Trajectories and Heatmap
        st.markdown("#### 🗺️ 2D Metric Pitch Tracking & Density Occupancy")
        fig_pitch = create_tracking_pitch_figure(cv_tracking)
        st.plotly_chart(fig_pitch, use_container_width=True)

        # Speed distribution
        st.markdown("#### ⚡ Physical Intensity & Velocity Profile")
        fig_spd = px.histogram(
            x=speeds,
            nbins=20,
            labels={"x": "Instantaneous Velocity (km/h)", "y": "Frequency"},
            title="Player Speed Distribution Across Match Sequence",
            template="plotly_dark",
            color_discrete_sequence=["#10B981"],
        )
        fig_spd.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.4)")
        st.plotly_chart(fig_spd, use_container_width=True)
    else:
        st.warning("No sample CV tracking JSON found at cv_module/samples/scout_tracking_poc.json")


# ==============================================================================
# TAB 5: MONEYBALL SQUAD BUILDER (MILP OPTIMIZER)
# ==============================================================================
with tabs[4]:
    st.subheader("🏆 Tactical Squad Builder (SciPy MILP 0-1 Knapsack)")
    st.write(
        "Solves the exact Mixed-Integer Linear Programming (MILP) optimization problem: "
        "Maximized team performance score subject to a hard budget cap, club limit quotas, and formation constraints."
    )

    s_c1, s_c2, s_c3 = st.columns(3)
    with s_c1:
        budget_cap_m = st.slider("Budget Cap (€M)", min_value=50.0, max_value=600.0, value=250.0, step=25.0)
    with s_c2:
        formation_choice = st.selectbox("Tactical Formation", options=["4-3-3", "4-2-3-1", "3-5-2"], index=0)
    with s_c3:
        max_club = st.slider("Max Players per Club", min_value=1, max_value=5, value=3)

    if st.button("⚡ Solve Optimal Squad Composition", type="primary"):
        with st.spinner("Solving MILP 0-1 Branch & Bound integer problem..."):
            res = squad_optimizer.optimize_squad(
                budget_cap_eur=budget_cap_m * 1_000_000.0,
                formation=formation_choice,
                max_per_club=max_club,
            )
            st.session_state["squad_result"] = res

    if "squad_result" in st.session_state:
        sq_res = st.session_state["squad_result"]
        st.markdown("---")

        sq_c1, sq_c2, sq_c3, sq_c4 = st.columns(4)
        with sq_c1:
            st.metric("Solver Status", "OPTIMAL (Solved)" if sq_res.get("success", False) else "FAILED")
        with sq_c2:
            st.metric("Total Squad Cost", f"€{sq_res['total_cost_eur']/1e6:.1f}M", f"Cap: €{budget_cap_m}M")
        with sq_c3:
            st.metric("Cumulative Performance Score", f"{sq_res['total_performance_score']:.1f}")
        with sq_c4:
            st.metric("Solver Compute Time", f"{sq_res.get('solve_time_seconds', 0.05):.3f}s")

        # Tactical Pitch Board
        fig_board = create_tactical_formation_board(sq_res["selected_players"], formation=formation_choice)
        st.plotly_chart(fig_board, use_container_width=True)

        # Squad Roster Table
        st.markdown("### 📋 Optimized 11 Roster")
        roster_rows = []
        for p in sq_res["selected_players"]:
            roster_rows.append({
                "Player": p["name"],
                "Club": p["club"],
                "League": p["league"],
                "Position": p["primary_position"],
                "Age": p["age"],
                "Cost (€M)": round(p["market_value_eur"] / 1e6, 2),
                "Perf Score": p.get("performance_score", 0),
            })
        st.dataframe(pd.DataFrame(roster_rows), use_container_width=True, hide_index=True)
