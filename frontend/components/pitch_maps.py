from typing import Dict, Any, List, Optional
import numpy as np
import plotly.graph_objects as go


def add_pitch_markings(fig: go.Figure) -> None:
    """
    Отрисовывает стандартные линии футбольного поля FIFA (105м x 68м) с повышенной контрастностью.
    """
    pitch_line_color = "rgba(255, 255, 255, 0.65)"
    pitch_line_width = 1.8

    # Внешняя граница и центральная линия
    fig.add_shape(type="rect", x0=0, y0=0, x1=105, y1=68,
                  line=dict(color=pitch_line_color, width=pitch_line_width))
    fig.add_shape(type="line", x0=52.5, y0=0, x1=52.5, y1=68,
                  line=dict(color=pitch_line_color, width=pitch_line_width))

    # Центральный круг (радиус 9.15м с центром в точке (52.5, 34))
    fig.add_shape(type="circle", x0=52.5 - 9.15, y0=34 - 9.15, x1=52.5 + 9.15, y1=34 + 9.15,
                  line=dict(color=pitch_line_color, width=pitch_line_width))
    # Центральная точка
    fig.add_trace(go.Scatter(x=[52.5], y=[34], mode="markers",
                             marker=dict(size=5, color="#FFFFFF"),
                             showlegend=False, hoverinfo="none"))

    # Левая штрафная и вратарская площадь
    fig.add_shape(type="rect", x0=0, y0=13.84, x1=16.5, y1=54.16,
                  line=dict(color=pitch_line_color, width=pitch_line_width))
    fig.add_shape(type="rect", x0=0, y0=24.84, x1=5.5, y1=43.16,
                  line=dict(color=pitch_line_color, width=pitch_line_width))
    fig.add_trace(go.Scatter(x=[11.0], y=[34.0], mode="markers",
                             marker=dict(size=5, color="#FFFFFF"),
                             showlegend=False, hoverinfo="none"))

    # Правая штрафная и вратарская площадь
    fig.add_shape(type="rect", x0=88.5, y0=13.84, x1=105, y1=54.16,
                  line=dict(color=pitch_line_color, width=pitch_line_width))
    fig.add_shape(type="rect", x0=99.5, y0=24.84, x1=105, y1=43.16,
                  line=dict(color=pitch_line_color, width=pitch_line_width))
    fig.add_trace(go.Scatter(x=[94.0], y=[34.0], mode="markers",
                             marker=dict(size=5, color="#FFFFFF"),
                             showlegend=False, hoverinfo="none"))

    # Ворота
    fig.add_shape(type="rect", x0=-2.0, y0=30.34, x1=0, y1=37.66,
                  line=dict(color="rgba(255,255,255,0.8)", width=1.5))
    fig.add_shape(type="rect", x0=105, y0=30.34, x1=107.0, y1=37.66,
                  line=dict(color="rgba(255,255,255,0.8)", width=1.5))


def create_tracking_pitch_figure(tracking_data: Dict[str, Any]) -> go.Figure:
    """
    Отрисовывает 2D-вид поля с траекториями игроков, перемещением мяча и тепловой картой плотности.
    """
    fig = go.Figure()
    add_pitch_markings(fig)

    frames = tracking_data.get("frames", [])
    if not frames:
        fig.update_layout(
            title="Нет данных трекинга",
            paper_bgcolor="#0B1120",
            plot_bgcolor="#0A2218",
            font=dict(color="#F8FAFC"),
        )
        return fig

    # Группировка координат по track_id
    track_coords: Dict[int, Dict[str, List]] = {}
    ball_coords: Dict[str, List] = {"x": [], "y": [], "frame": []}
    all_x, all_y = [], []

    for f in frames:
        f_idx = f.get("frame_index", 0)
        # Игроки
        for p in f.get("players", []):
            tid = p.get("track_id", -1)
            if "pitch_pos_meters" in p:
                x, y = p["pitch_pos_meters"]
            else:
                x = p.get("pitch_x", 0.0)
                y = p.get("pitch_y", 0.0)
            speed = p.get("speed_kmh", 0.0)

            if tid not in track_coords:
                track_coords[tid] = {"x": [], "y": [], "speed": [], "frame": []}
            track_coords[tid]["x"].append(x)
            track_coords[tid]["y"].append(y)
            track_coords[tid]["speed"].append(speed)
            track_coords[tid]["frame"].append(f_idx)
            all_x.append(x)
            all_y.append(y)

        # Мяч
        ball = f.get("ball")
        if ball:
            if "pitch_pos_meters" in ball:
                bx, by = ball["pitch_pos_meters"]
            else:
                bx = ball.get("pitch_x", 0.0)
                by = ball.get("pitch_y", 0.0)
            ball_coords["x"].append(bx)
            ball_coords["y"].append(by)
            ball_coords["frame"].append(f_idx)

    # 2D-контур плотности позиций
    if len(all_x) > 10:
        fig.add_trace(
            go.Histogram2dContour(
                x=all_x,
                y=all_y,
                colorscale="Viridis",
                reversescale=False,
                opacity=0.4,
                showscale=False,
                contours=dict(showlines=False),
                hoverinfo="none",
                name="Плотность игроков",
            )
        )

    # Траектории игроков (яркие различимые цвета)
    palette = ["#38BDF8", "#34D399", "#F472B6", "#A78BFA", "#FBBF24", "#FB923C", "#F87171", "#2DD4BF"]
    for i, (tid, data) in enumerate(track_coords.items()):
        color = palette[i % len(palette)]
        avg_speed = np.mean(data["speed"]) if data["speed"] else 0.0
        fig.add_trace(
            go.Scatter(
                x=data["x"],
                y=data["y"],
                mode="lines+markers",
                line=dict(color=color, width=2.5),
                marker=dict(size=8, color=color, symbol="circle", line=dict(color="#FFFFFF", width=1)),
                name=f"Игрок #{tid} (Ср. {avg_speed:.1f} км/ч)",
                text=[f"Игрок #{tid}<br>Кадр: {fr}<br>Скорость: {sp:.1f} км/ч" for fr, sp in zip(data["frame"], data["speed"])],
                hoverinfo="text",
            )
        )

    # Траектория мяча
    if ball_coords["x"]:
        fig.add_trace(
            go.Scatter(
                x=ball_coords["x"],
                y=ball_coords["y"],
                mode="lines+markers",
                line=dict(color="#FFD700", width=3, dash="dot"),
                marker=dict(size=10, color="#FFFFFF", line=dict(color="#FFD700", width=2.5)),
                name="⚽ Мяч",
                text=[f"Координата мяча<br>Кадр: {fr}" for fr in ball_coords["frame"]],
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title="<b>CV 2D-трекинг координат и тепловая карта на поле FIFA (105м x 68м)</b>",
        template="plotly_dark",
        plot_bgcolor="#0A2218",  # Глубокий тактический зеленый газон
        paper_bgcolor="#0B1120",
        font=dict(family="Inter, sans-serif", color="#F8FAFC"),
        xaxis=dict(
            range=[-4, 109],
            showgrid=False,
            zeroline=False,
            title="Длина поля (метры)",
            color="#E2E8F0",
            tickfont=dict(color="#CBD5E1"),
        ),
        yaxis=dict(
            range=[-4, 72],
            showgrid=False,
            zeroline=False,
            scaleanchor="x",
            scaleratio=1,
            title="Ширина поля (метры)",
            color="#E2E8F0",
            tickfont=dict(color="#CBD5E1"),
        ),
        height=540,
        margin=dict(l=30, r=30, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
            font=dict(size=11, color="#F8FAFC"),
            bgcolor="rgba(15, 23, 42, 0.85)",
            bordercolor="rgba(255, 255, 255, 0.2)",
            borderwidth=1,
        ),
    )

    return fig


def create_tactical_formation_board(squad_players: List[Dict[str, Any]], formation: str = "4-3-3") -> go.Figure:
    """
    Отрисовывает тактический планшет с 11 игроками стартового состава по амплуа и расстановке.
    """
    fig = go.Figure()
    add_pitch_markings(fig)

    coord_slots = {
        "4-3-3": [
            ("ВРТ", 8.0, 34.0),
            ("ЛЗ", 28.0, 10.0),
            ("ЛЦЗ", 24.0, 26.0),
            ("ПЦЗ", 24.0, 42.0),
            ("ПЗ", 28.0, 58.0),
            ("ОП", 48.0, 34.0),
            ("ЛЦП", 56.0, 20.0),
            ("ПЦП", 56.0, 48.0),
            ("ЛВ", 78.0, 12.0),
            ("ФРВ", 86.0, 34.0),
            ("ПВ", 78.0, 56.0),
        ],
        "4-2-3-1": [
            ("ВРТ", 8.0, 34.0),
            ("ЛЗ", 28.0, 10.0),
            ("ЛЦЗ", 24.0, 26.0),
            ("ПЦЗ", 24.0, 42.0),
            ("ПЗ", 28.0, 58.0),
            ("ЛОП", 45.0, 24.0),
            ("РОП", 45.0, 44.0),
            ("ЦАП", 66.0, 34.0),
            ("ЛАП", 68.0, 14.0),
            ("ПАП", 68.0, 54.0),
            ("ФРВ", 88.0, 34.0),
        ],
        "3-5-2": [
            ("ВРТ", 8.0, 34.0),
            ("ЛЦЗ", 24.0, 16.0),
            ("ЦЗ", 22.0, 34.0),
            ("ПЦЗ", 24.0, 52.0),
            ("ЛЛЗ", 46.0, 8.0),
            ("ЛОП", 46.0, 26.0),
            ("РОП", 46.0, 42.0),
            ("ПЛЗ", 46.0, 60.0),
            ("ЦАП", 64.0, 34.0),
            ("ЛФРВ", 84.0, 24.0),
            ("ПФРВ", 84.0, 44.0),
        ],
    }

    slots = coord_slots.get(formation, coord_slots["4-3-3"])

    xs, ys, texts, labels, colors = [], [], [], [], []
    for i, p in enumerate(squad_players[:11]):
        if i >= len(slots):
            break
        role, sx, sy = slots[i]
        name = p.get("name", f"Игрок #{i+1}")
        pos = p.get("primary_position", role)
        club = p.get("club", "Свободный агент")
        val = p.get("market_value_eur", 0) / 1_000_000.0

        xs.append(sx)
        ys.append(sy)
        labels.append(name.split()[-1] if name else f"И{i+1}")
        texts.append(
            f"<b>{name}</b> ({pos})<br>"
            f"Клуб: {club}<br>"
            f"Позиция на схеме: {role}<br>"
            f"Рыночная стоимость: €{val:.1f} млн"
        )
        if pos == "GK" or role == "ВРТ":
            colors.append("#F59E0B")  # Янтарный для вратаря
        elif "CB" in pos or "B" in pos or "З" in role:
            colors.append("#3B82F6")  # Синий для защиты
        elif "M" in pos or "П" in role:
            colors.append("#10B981")  # Изумрудный для полузащиты
        else:
            colors.append("#EC4899")  # Розово-красный для нападения

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            marker=dict(
                size=34,
                color=colors,
                line=dict(color="#FFFFFF", width=2.5),
                symbol="circle",
            ),
            text=labels,
            textposition="bottom center",
            textfont=dict(color="#FFFFFF", size=12, family="Inter, sans-serif"),
            hovertext=texts,
            hoverinfo="text",
            showlegend=False,
        )
    )

    fig.update_layout(
        title=f"<b>Тактическая расстановка XI (Схема {formation})</b>",
        template="plotly_dark",
        plot_bgcolor="#0A2218",
        paper_bgcolor="#0B1120",
        font=dict(family="Inter, sans-serif", color="#F8FAFC"),
        xaxis=dict(range=[-3, 108], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[-3, 71], showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1),
        height=540,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig
