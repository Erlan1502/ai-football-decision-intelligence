from typing import List, Dict, Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_valuation_scatter_matrix(players: List[Dict[str, Any]]) -> go.Figure:
    """
    Формирует интерактивную матрицу трансферного арбитража Moneyball 2.0.
    Сопоставляет рыночную стоимость Transfermarkt и модельную справедливую оценку.
    """
    rows = []
    for p in players:
        val = p.get("valuation", {})
        pred_val = val.get("predicted_value_eur", p["market_value_eur"])
        delta = val.get("undervaluation_delta_eur", pred_val - p["market_value_eur"])
        ratio = val.get("undervaluation_ratio", pred_val / (p["market_value_eur"] + 1))

        rows.append({
            "Игрок": p["name"],
            "Клуб": p["club"],
            "Лига": p["league"],
            "Позиция": p["primary_position"],
            "Возраст": p["age"],
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
        fig.update_layout(
            title="Нет игроков, соответствующих выбранным фильтрам.",
            paper_bgcolor="#0B1120",
            plot_bgcolor="#1E293B",
            font=dict(color="#F8FAFC"),
        )
        return fig

    max_val = max(df["market_value_eur"].max(), df["predicted_value_eur"].max()) * 1.05

    fig = px.scatter(
        df,
        x="market_value_eur",
        y="predicted_value_eur",
        color="delta_millions",
        size="performance_score",
        hover_name="Игрок",
        hover_data={
            "Клуб": True,
            "Лига": True,
            "Позиция": True,
            "Возраст": True,
            "market_value_eur": ":,.0f",
            "predicted_value_eur": ":,.0f",
            "delta_millions": ":+.1f",
            "performance_score": ":.1f",
        },
        color_continuous_scale="Turbo",
        labels={
            "market_value_eur": "Рыночная цена Transfermarkt (€)",
            "predicted_value_eur": "Справедливая цена Moneyball 2.0 (€)",
            "delta_millions": "Дельта арбитража (€ млн)",
            "performance_score": "Индекс эффективности",
        },
        title="<b>Матрица трансферного арбитража Moneyball 2.0</b> (Точки выше диагонали — недооцененные игроки)",
    )

    # Reference line y = x (Fair Market Valuation)
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line=dict(color="rgba(255, 255, 255, 0.75)", dash="dash", width=2),
            name="Базовая линия справедливости (y = x)",
            hoverinfo="none",
        )
    )

    fig.update_traces(
        marker=dict(line=dict(width=1, color="rgba(255, 255, 255, 0.5)")),
    )

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#131D31",
        paper_bgcolor="#0B1120",
        font=dict(family="Inter, sans-serif", color="#F8FAFC", size=12),
        height=620,
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.12)",
            tickfont=dict(color="#CBD5E1", size=11),
            title=dict(text="Рыночная цена Transfermarkt (€)", font=dict(color="#F8FAFC", size=13)),
            tickprefix="€",
            tickformat=",.0s",
        ),
        yaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.12)",
            tickfont=dict(color="#CBD5E1", size=11),
            title=dict(text="Справедливая оценка Moneyball 2.0 (€)", font=dict(color="#F8FAFC", size=13)),
            tickprefix="€",
            tickformat=",.0s",
        ),
        coloraxis_colorbar=dict(
            title=dict(text="Дельта (€ млн)", font=dict(color="#F8FAFC", size=12)),
            tickfont=dict(color="#CBD5E1", size=11),
        ),
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(
            font=dict(color="#F8FAFC", size=12),
            bgcolor="rgba(15, 23, 42, 0.8)",
            bordercolor="rgba(255, 255, 255, 0.2)",
            borderwidth=1,
        ),
    )

    return fig
