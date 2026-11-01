from typing import List, Dict, Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_valuation_scatter_matrix(players: List[Dict[str, Any]]) -> go.Figure:
    """
    Renders the Executive Moneyball Valuation Matrix.
    Plots Market Value vs Model Predicted Value with Moneyball Delta color coding.
    """
    rows = []
    for p in players:
        val = p.get("valuation", {})
        pred_val = val.get("predicted_value_eur", p["market_value_eur"])
        delta = val.get("undervaluation_delta_eur", pred_val - p["market_value_eur"])
        ratio = val.get("undervaluation_ratio", pred_val / (p["market_value_eur"] + 1))

        rows.append({
            "name": p["name"],
            "club": p["club"],
            "league": p["league"],
            "position": p["primary_position"],
            "age": p["age"],
            "market_value_eur": p["market_value_eur"],
            "predicted_value_eur": pred_val,
            "delta_eur": delta,
            "delta_millions": round(delta / 1_000_000.0, 1),
            "ratio": ratio,
            "performance_score": p.get("performance_score", 65.0),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No players match the current filters.")
        return fig

    max_val = max(df["market_value_eur"].max(), df["predicted_value_eur"].max()) * 1.05

    fig = px.scatter(
        df,
        x="market_value_eur",
        y="predicted_value_eur",
        color="delta_millions",
        size="performance_score",
        hover_name="name",
        hover_data={
            "club": True,
            "league": True,
            "position": True,
            "age": True,
            "market_value_eur": ":,.0f",
            "predicted_value_eur": ":,.0f",
            "delta_millions": ":+.1f",
            "performance_score": True,
        },
        color_continuous_scale="Viridis",
        labels={
            "market_value_eur": "Transfermarkt Price (€)",
            "predicted_value_eur": "Moneyball 2.0 Fair Value (€)",
            "delta_millions": "Moneyball Delta (€M)",
            "performance_score": "Performance Score",
        },
        title="<b>Moneyball 2.0 Market Arbitrage Matrix</b> (Points above diagonal = Undervalued Gems)",
    )

    # Add Fair Value Reference Line (y = x)
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line=dict(color="rgba(255, 255, 255, 0.4)", dash="dash", width=2),
            name="Fair Market Valuation Baseline",
            hoverinfo="none",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        paper_bgcolor="rgba(15, 23, 42, 0.0)",
        font=dict(family="Inter, sans-serif", color="#F8FAFC"),
        height=620,
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)", tickprefix="€", tickformat=",.0s"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", tickprefix="€", tickformat=",.0s"),
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return fig
