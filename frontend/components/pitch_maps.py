from typing import Dict, Any, List, Optional
import numpy as np
import plotly.graph_objects as go


def add_pitch_markings(fig: go.Figure) -> None:
    """
    Draws professional FIFA standard pitch lines (105m x 68m) on a Plotly figure.
    """
    pitch_line_color = "rgba(255, 255, 255, 0.4)"
    pitch_line_width = 1.5

    # Boundary and Halfway line
    fig.add_shape(type="rect", x0=0, y0=0, x1=105, y1=68,
                  line=dict(color=pitch_line_color, width=pitch_line_width))
    fig.add_shape(type="line", x0=52.5, y0=0, x1=52.5, y1=68,
                  line=dict(color=pitch_line_color, width=pitch_line_width))

    # Center circle (radius 9.15m at (52.5, 34))
    fig.add_shape(type="circle", x0=52.5 - 9.15, y0=34 - 9.15, x1=52.5 + 9.15, y1=34 + 9.15,
                  line=dict(color=pitch_line_color, width=pitch_line_width))
    # Center spot
    fig.add_trace(go.Scatter(x=[52.5], y=[34], mode="markers",
                             marker=dict(size=4, color="white"),
                             showlegend=False, hoverinfo="none"))

    # Left Penalty box & 6-yard box
    fig.add_shape(type="rect", x0=0, y0=13.84, x1=16.5, y1=54.16,
                  line=dict(color=pitch_line_color, width=pitch_line_width))
    fig.add_shape(type="rect", x0=0, y0=24.84, x1=5.5, y1=43.16,
                  line=dict(color=pitch_line_color, width=pitch_line_width))
    fig.add_trace(go.Scatter(x=[11.0], y=[34.0], mode="markers",
                             marker=dict(size=4, color="white"),
                             showlegend=False, hoverinfo="none"))

    # Right Penalty box & 6-yard box
    fig.add_shape(type="rect", x0=88.5, y0=13.84, x1=105, y1=54.16,
                  line=dict(color=pitch_line_color, width=pitch_line_width))
    fig.add_shape(type="rect", x0=99.5, y0=24.84, x1=105, y1=43.16,
                  line=dict(color=pitch_line_color, width=pitch_line_width))
    fig.add_trace(go.Scatter(x=[94.0], y=[34.0], mode="markers",
                             marker=dict(size=4, color="white"),
                             showlegend=False, hoverinfo="none"))

    # Goal frames
    fig.add_shape(type="rect", x0=-2.0, y0=30.34, x1=0, y1=37.66,
                  line=dict(color="rgba(255,255,255,0.6)", width=1))
    fig.add_shape(type="rect", x0=105, y0=30.34, x1=107.0, y1=37.66,
                  line=dict(color="rgba(255,255,255,0.6)", width=1))


def create_tracking_pitch_figure(tracking_data: Dict[str, Any]) -> go.Figure:
    """
    Renders 2D pitch view with player trajectories, ball tracking, and heat contours.
    """
    fig = go.Figure()
    add_pitch_markings(fig)

    frames = tracking_data.get("frames", [])
    if not frames:
        fig.update_layout(title="No tracking frames available")
        return fig

    # Group coordinates by track_id
    track_coords: Dict[int, Dict[str, List]] = {}
    ball_coords: Dict[str, List] = {"x": [], "y": [], "frame": []}
    all_x, all_y = [], []

    for f in frames:
        f_idx = f.get("frame_index", 0)
        # Players
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

        # Ball
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

    # Add 2D Density Heatmap Contour if enough points
    if len(all_x) > 10:
        fig.add_trace(
            go.Histogram2dContour(
                x=all_x,
                y=all_y,
                colorscale="Viridis",
                reversescale=False,
                opacity=0.35,
                showscale=False,
                contours=dict(showlines=False),
                hoverinfo="none",
                name="Occupancy Density",
            )
        )

    # Plot Player Trajectories
    palette = ["#38BDF8", "#34D399", "#F472B6", "#A78BFA", "#FBBF24", "#FB923C", "#F87171", "#2DD4BF"]
    for i, (tid, data) in enumerate(track_coords.items()):
        color = palette[i % len(palette)]
        avg_speed = np.mean(data["speed"]) if data["speed"] else 0.0
        # Line trajectory
        fig.add_trace(
            go.Scatter(
                x=data["x"],
                y=data["y"],
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=7, color=color, symbol="circle"),
                name=f"Player #{tid} (Avg {avg_speed:.1f} km/h)",
                text=[f"Player #{tid}<br>Frame {fr}<br>Speed: {sp:.1f} km/h" for fr, sp in zip(data["frame"], data["speed"])],
                hoverinfo="text",
            )
        )

    # Plot Ball Trajectory
    if ball_coords["x"]:
        fig.add_trace(
            go.Scatter(
                x=ball_coords["x"],
                y=ball_coords["y"],
                mode="lines+markers",
                line=dict(color="#FFD700", width=3, dash="dot"),
                marker=dict(size=9, color="#FFFFFF", line=dict(color="#FFD700", width=2)),
                name="⚽ Match Ball",
                text=[f"Ball Pos<br>Frame {fr}" for fr in ball_coords["frame"]],
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title="<b>CV 2D Pitch Coordinate Tracking & Position Heatmap</b> (105m x 68m FIFA Pitch)",
        template="plotly_dark",
        plot_bgcolor="#0A2218",  # Tactical deep grass green
        paper_bgcolor="rgba(15, 23, 42, 0.0)",
        font=dict(family="Inter, sans-serif", color="#F8FAFC"),
        xaxis=dict(
            range=[-4, 109],
            showgrid=False,
            zeroline=False,
            title="Pitch Length (Meters)",
            color="rgba(255,255,255,0.7)",
        ),
        yaxis=dict(
            range=[-4, 72],
            showgrid=False,
            zeroline=False,
            scaleanchor="x",
            scaleratio=1,
            title="Pitch Width (Meters)",
            color="rgba(255,255,255,0.7)",
        ),
        height=540,
        margin=dict(l=30, r=30, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
    )

    return fig


def create_tactical_formation_board(squad_players: List[Dict[str, Any]], formation: str = "4-3-3") -> go.Figure:
    """
    Renders an executive tactical pitch board with 11 players in their assigned roles.
    """
    fig = go.Figure()
    add_pitch_markings(fig)

    # Canonical coordinate templates for formations on 105 x 68 pitch (playing left-to-right)
    # Roles map to target (x, y) coordinates
    coord_slots = {
        "4-3-3": [
            ("GK", 8.0, 34.0),
            ("LB", 28.0, 10.0),
            ("LCB", 24.0, 26.0),
            ("RCB", 24.0, 42.0),
            ("RB", 28.0, 58.0),
            ("DM", 48.0, 34.0),
            ("LCM", 56.0, 20.0),
            ("RCM", 56.0, 48.0),
            ("LW", 78.0, 12.0),
            ("ST", 86.0, 34.0),
            ("RW", 78.0, 56.0),
        ],
        "4-2-3-1": [
            ("GK", 8.0, 34.0),
            ("LB", 28.0, 10.0),
            ("LCB", 24.0, 26.0),
            ("RCB", 24.0, 42.0),
            ("RB", 28.0, 58.0),
            ("LDM", 45.0, 24.0),
            ("RDM", 45.0, 44.0),
            ("CAM", 66.0, 34.0),
            ("LAM", 68.0, 14.0),
            ("RAM", 68.0, 54.0),
            ("ST", 88.0, 34.0),
        ],
    }

    slots = coord_slots.get(formation, coord_slots["4-3-3"])

    xs, ys, texts, labels, colors = [], [], [], [], []
    for i, p in enumerate(squad_players[:11]):
        if i >= len(slots):
            break
        role, sx, sy = slots[i]
        name = p.get("name", f"Player #{i+1}")
        pos = p.get("primary_position", role)
        club = p.get("club", "Free Agent")
        val = p.get("market_value_eur", 0) / 1_000_000.0

        xs.append(sx)
        ys.append(sy)
        labels.append(name.split()[-1] if name else f"P{i+1}")
        texts.append(
            f"<b>{name}</b> ({pos})<br>"
            f"Club: {club}<br>"
            f"Assigned Role: {role}<br>"
            f"Market Value: €{val:.1f}M"
        )
        if pos == "GK":
            colors.append("#F59E0B")  # Amber for keeper
        elif "CB" in pos or "B" in pos:
            colors.append("#3B82F6")  # Blue for defense
        elif "M" in pos:
            colors.append("#10B981")  # Green for midfield
        else:
            colors.append("#EC4899")  # Pink/Red for attack

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            marker=dict(
                size=32,
                color=colors,
                line=dict(color="#FFFFFF", width=2),
                symbol="circle",
            ),
            text=labels,
            textposition="bottom center",
            textfont=dict(color="#FFFFFF", size=11, family="Inter, sans-serif"),
            hovertext=texts,
            hoverinfo="text",
            showlegend=False,
        )
    )

    fig.update_layout(
        title=f"<b>Tactical XI Pitch Board ({formation} Formation)</b>",
        template="plotly_dark",
        plot_bgcolor="#0E2E20",
        paper_bgcolor="rgba(15, 23, 42, 0.0)",
        font=dict(family="Inter, sans-serif", color="#F8FAFC"),
        xaxis=dict(range=[-3, 108], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[-3, 71], showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1),
        height=540,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig
