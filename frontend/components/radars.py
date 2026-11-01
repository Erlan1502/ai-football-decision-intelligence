from typing import Dict, Any, List, Optional
import plotly.graph_objects as go
import numpy as np


def compute_standardized_radar_attributes(player: Dict[str, Any]) -> Dict[str, float]:
    """
    Рассчитывает 6 нормированных тактических столпов (от 0 до 100) из метрик p90
    для сбалансированного, наглядного и высококонтрастного отображения радарных графиков.
    """
    m = player.get("metrics") or player

    # 1. Продвижение мяча: prog_passes_p90 (0..10) и prog_carries_p90 (0..6)
    prog_passes = float(m.get("prog_passes_p90", 3.0))
    prog_carries = float(m.get("prog_carries_p90", 2.0))
    progression_score = min(100.0, max(10.0, (prog_passes / 8.5) * 60.0 + (prog_carries / 5.0) * 40.0))

    # 2. Создание моментов: key_passes_p90 (0..3.5), xa_p90 (0..0.45), passes_into_box (0..3)
    key_passes = float(m.get("key_passes_p90", 1.0))
    xa = float(m.get("xa_p90", 0.12))
    passes_box = float(m.get("passes_into_box_p90", 1.0))
    creation_score = min(100.0, max(10.0, (key_passes / 3.0) * 40.0 + (xa / 0.4) * 40.0 + (passes_box / 3.0) * 20.0))

    # 3. Ударная угроза: npxg_p90 (0..0.7), shots_p90 (0..4.5), goals_p90 (0..0.5)
    npxg = float(m.get("npxg_p90", 0.15))
    shots = float(m.get("shots_p90", 1.5))
    goals = float(m.get("goals_p90", 0.1))
    threat_score = min(100.0, max(10.0, (npxg / 0.65) * 50.0 + (shots / 4.0) * 30.0 + (goals / 0.5) * 20.0))

    # 4. Контроль и дриблинг: pass_completion_pct (65..95), take_ons_won_pct (30..80)
    pass_acc = float(m.get("pass_completion_pct", 80.0))
    dribble_acc = float(m.get("take_ons_won_pct", 50.0))
    retention_score = min(100.0, max(10.0, ((pass_acc - 65.0) / 28.0) * 60.0 + ((dribble_acc - 30.0) / 45.0) * 40.0))

    # 5. Оборонительная работа: tackles_interceptions_p90 (0.5..5.5), ball_recoveries_p90 (2..10)
    def_actions = float(m.get("tackles_interceptions_p90", 2.0))
    recoveries = float(m.get("ball_recoveries_p90", 5.0))
    defense_score = min(100.0, max(10.0, (def_actions / 5.0) * 55.0 + (recoveries / 9.0) * 45.0))

    # 6. Борьба в воздухе: aerials_won_pct (20..80)
    aerial_acc = float(m.get("aerials_won_pct", 50.0))
    aerial_score = min(100.0, max(10.0, ((aerial_acc - 20.0) / 60.0) * 100.0))

    return {
        "Продвижение мяча": round(progression_score, 1),
        "Создание моментов": round(creation_score, 1),
        "Ударная угроза": round(threat_score, 1),
        "Контроль и дриблинг": round(retention_score, 1),
        "Оборонительная работа": round(defense_score, 1),
        "Борьба в воздухе": round(aerial_score, 1),
    }


def create_radar_comparison(
    target_player: Dict[str, Any],
    clone_player: Dict[str, Any],
    similarity_score: Optional[float] = None,
) -> go.Figure:
    """
    Отрисовывает радарное сравнение тактического профиля эталонного игрока и найденного аналога.
    """
    target_scores = compute_standardized_radar_attributes(target_player)
    clone_scores = compute_standardized_radar_attributes(clone_player)

    categories = list(target_scores.keys())
    # Замыкаем контур для построения замкнутого радара
    categories_closed = categories + [categories[0]]
    r_target = [target_scores[c] for c in categories] + [target_scores[categories[0]]]
    r_clone = [clone_scores[c] for c in categories] + [clone_scores[categories[0]]]

    target_name = target_player.get("name", "Эталонный игрок")
    clone_name = clone_player.get("name", "Кандидат-аналог")
    target_val = target_player.get("market_value_eur", 0) / 1_000_000.0
    clone_val = clone_player.get("market_value_eur", 0) / 1_000_000.0

    title_text = f"<b>Тактическое DNA-сопоставление: {target_name} vs {clone_name}</b>"
    if similarity_score is not None:
        title_text += f" (Сходство: {similarity_score * 100:.1f}%)"

    fig = go.Figure()

    # Эталонный игрок (Яркий бирюзовый / Cyan)
    fig.add_trace(
        go.Scatterpolar(
            r=r_target,
            theta=categories_closed,
            fill="toself",
            name=f"{target_name} (€{target_val:.1f}M — Эталон)",
            line=dict(color="#00E5FF", width=3),
            fillcolor="rgba(0, 229, 255, 0.22)",
            marker=dict(size=7, color="#00E5FF"),
        )
    )

    # Кандидат-аналог (Яркий изумрудный / Emerald)
    fig.add_trace(
        go.Scatterpolar(
            r=r_clone,
            theta=categories_closed,
            fill="toself",
            name=f"{clone_name} (€{clone_val:.1f}M — Арбитражный аналог)",
            line=dict(color="#10B981", width=3),
            fillcolor="rgba(16, 185, 129, 0.35)",
            marker=dict(size=7, color="#10B981"),
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor="rgba(255, 255, 255, 0.18)",
                linecolor="rgba(255, 255, 255, 0.35)",
                tickfont=dict(color="#CBD5E1", size=11),
            ),
            angularaxis=dict(
                gridcolor="rgba(255, 255, 255, 0.18)",
                linecolor="rgba(255, 255, 255, 0.35)",
                tickfont=dict(color="#FFFFFF", size=12, family="Inter, sans-serif"),
            ),
            bgcolor="#131D31",
        ),
        title=dict(
            text=title_text,
            font=dict(family="Inter, sans-serif", size=16, color="#F8FAFC"),
            x=0.05,
        ),
        template="plotly_dark",
        paper_bgcolor="#0B1120",
        plot_bgcolor="#0B1120",
        height=520,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.16,
            xanchor="center",
            x=0.5,
            font=dict(color="#F8FAFC", size=12),
            bgcolor="rgba(15, 23, 42, 0.8)",
            bordercolor="rgba(255, 255, 255, 0.2)",
            borderwidth=1,
        ),
        margin=dict(l=60, r=60, t=60, b=60),
    )

    return fig
