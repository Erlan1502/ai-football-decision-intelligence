from typing import Dict, Any, List, Optional
import plotly.graph_objects as go
import numpy as np


def compute_standardized_radar_attributes(player: Dict[str, Any]) -> Dict[str, float]:
    """
    Computes 6 normalized tactical pillars (0 to 100) from p90 metrics
    to ensure radar charts are realistic, balanced, and instantly readable.
    """
    m = player.get("metrics") or player

    # 1. Progression: based on prog_passes_p90 (0 to 10) and prog_carries_p90 (0 to 6)
    prog_passes = float(m.get("prog_passes_p90", 3.0))
    prog_carries = float(m.get("prog_carries_p90", 2.0))
    progression_score = min(100.0, max(10.0, (prog_passes / 8.5) * 60.0 + (prog_carries / 5.0) * 40.0))

    # 2. Chance Creation: key_passes_p90 (0 to 3.5), xa_p90 (0 to 0.45)
    key_passes = float(m.get("key_passes_p90", 1.0))
    xa = float(m.get("xa_p90", 0.12))
    passes_box = float(m.get("passes_into_box_p90", 1.0))
    creation_score = min(100.0, max(10.0, (key_passes / 3.0) * 40.0 + (xa / 0.4) * 40.0 + (passes_box / 3.0) * 20.0))

    # 3. Goal Threat: npxg_p90 (0 to 0.7), shots_p90 (0 to 4.5)
    npxg = float(m.get("npxg_p90", 0.15))
    shots = float(m.get("shots_p90", 1.5))
    goals = float(m.get("goals_p90", 0.1))
    threat_score = min(100.0, max(10.0, (npxg / 0.65) * 50.0 + (shots / 4.0) * 30.0 + (goals / 0.5) * 20.0))

    # 4. Ball Retention & Dribbling: pass_completion_pct (65-95), take_ons_won_pct (30-80)
    pass_acc = float(m.get("pass_completion_pct", 80.0))
    dribble_acc = float(m.get("take_ons_won_pct", 50.0))
    retention_score = min(100.0, max(10.0, ((pass_acc - 65.0) / 28.0) * 60.0 + ((dribble_acc - 30.0) / 45.0) * 40.0))

    # 5. Defensive Workrate: tackles_interceptions_p90 (0.5 to 5.5), ball_recoveries_p90 (2 to 10)
    def_actions = float(m.get("tackles_interceptions_p90", 2.0))
    recoveries = float(m.get("ball_recoveries_p90", 5.0))
    defense_score = min(100.0, max(10.0, (def_actions / 5.0) * 55.0 + (recoveries / 9.0) * 45.0))

    # 6. Physical & Aerial: aerials_won_pct (20 to 80)
    aerial_acc = float(m.get("aerials_won_pct", 50.0))
    aerial_score = min(100.0, max(10.0, ((aerial_acc - 20.0) / 60.0) * 100.0))

    return {
        "Progression": round(progression_score, 1),
        "Chance Creation": round(creation_score, 1),
        "Goal Threat": round(threat_score, 1),
        "Ball Retention": round(retention_score, 1),
        "Defensive Work": round(defense_score, 1),
        "Aerial Duels": round(aerial_score, 1),
    }


def create_radar_comparison(
    target_player: Dict[str, Any],
    clone_player: Dict[str, Any],
    similarity_score: Optional[float] = None,
) -> go.Figure:
    """
    Renders an elite radar chart comparing Target Star and Identified Analog Clone.
    """
    target_scores = compute_standardized_radar_attributes(target_player)
    clone_scores = compute_standardized_radar_attributes(clone_player)

    categories = list(target_scores.keys())
    # Closed loop for radar chart
    categories_closed = categories + [categories[0]]
    r_target = [target_scores[c] for c in categories] + [target_scores[categories[0]]]
    r_clone = [clone_scores[c] for c in categories] + [clone_scores[categories[0]]]

    target_name = target_player.get("name", "Target Star")
    clone_name = clone_player.get("name", "Clone Candidate")
    target_val = target_player.get("market_value_eur", 0) / 1_000_000.0
    clone_val = clone_player.get("market_value_eur", 0) / 1_000_000.0

    title_text = f"<b>Tactical DNA Match: {target_name} vs {clone_name}</b>"
    if similarity_score is not None:
        title_text += f" (Cosine Similarity: {similarity_score * 100:.1f}%)"

    fig = go.Figure()

    # Target Star Profile (Electric Blue / Cyan)
    fig.add_trace(
        go.Scatterpolar(
            r=r_target,
            theta=categories_closed,
            fill="toself",
            name=f"{target_name} (€{target_val:.1f}M)",
            line=dict(color="#00D2FF", width=2.5),
            fillcolor="rgba(0, 210, 255, 0.25)",
            marker=dict(size=6, color="#00D2FF"),
        )
    )

    # Clone Candidate Profile (Neon Amber / Green)
    fig.add_trace(
        go.Scatterpolar(
            r=r_clone,
            theta=categories_closed,
            fill="toself",
            name=f"{clone_name} (€{clone_val:.1f}M - Arbitrage Clone)",
            line=dict(color="#10B981", width=2.5),
            fillcolor="rgba(16, 185, 129, 0.35)",
            marker=dict(size=6, color="#10B981"),
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor="rgba(255, 255, 255, 0.12)",
                linecolor="rgba(255, 255, 255, 0.2)",
                tickfont=dict(color="rgba(255, 255, 255, 0.6)", size=10),
            ),
            angularaxis=dict(
                gridcolor="rgba(255, 255, 255, 0.12)",
                linecolor="rgba(255, 255, 255, 0.2)",
                tickfont=dict(color="#F8FAFC", size=12, family="Inter, sans-serif"),
            ),
            bgcolor="rgba(15, 23, 42, 0.4)",
        ),
        title=dict(
            text=title_text,
            font=dict(family="Inter, sans-serif", size=16, color="#F8FAFC"),
            x=0.05,
        ),
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.0)",
        plot_bgcolor="rgba(15, 23, 42, 0.0)",
        height=520,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(color="#F8FAFC", size=12),
        ),
        margin=dict(l=60, r=60, t=60, b=60),
    )

    return fig
