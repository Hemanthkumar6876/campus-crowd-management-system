"""Heatmaps and hotspot detection (image level and campus level)."""
import cv2
import numpy as np
import plotly.graph_objects as go

from config import LEVEL_COLORS, LOCATIONS, classify

PLOTLY_SCALE = [[0, "#1a9850"], [0.25, "#a6d96a"], [0.5, "#fee08b"], [0.75, "#f46d43"], [1, "#a50026"]]


# --------------------------- image level ---------------------------------
def person_points(boxes):
    """Approximate ground position of each person (centre-bottom of the box)."""
    pts = []
    for x1, y1, x2, y2, *_ in boxes:
        pts.append(((x1 + x2) / 2, y2 - 0.15 * (y2 - y1)))
    return pts


def density_map(shape, boxes, sigma_frac=0.04):
    h, w = shape[:2]
    m = np.zeros((h, w), np.float32)
    for x, y in person_points(boxes):
        xi, yi = int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))
        m[yi, xi] += 1
    sigma = max(h, w) * sigma_frac
    return cv2.GaussianBlur(m, (0, 0), sigma) if m.any() else m


def overlay_heatmap(bgr, dmap, alpha=0.55):
    if dmap.max() <= 0:
        return bgr.copy()
    norm = (dmap / dmap.max() * 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    mask = (dmap / dmap.max() > 0.05)[..., None]
    blended = cv2.addWeighted(bgr, 1 - alpha, color, alpha, 0)
    return np.where(mask, blended, bgr)


def find_hotspots(dmap, boxes, rel_thresh=0.55, min_people=3):
    """Connected high-density regions, ranked by the number of people inside."""
    if dmap.max() <= 0:
        return []
    binary = (dmap >= rel_thresh * dmap.max()).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary)
    pts = person_points(boxes)
    spots = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        inside = sum(1 for px, py in pts if x <= px <= x + w and y <= py <= y + h)
        if inside >= min_people:
            spots.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "people": inside})
    spots.sort(key=lambda s: s["people"], reverse=True)
    total = max(len(boxes), 1)
    for r, s in enumerate(spots, 1):
        s["rank"] = r
        s["share_pct"] = round(100 * s["people"] / total, 1)
    return spots


def draw_boxes(bgr, boxes, color=(60, 220, 60)):
    out = bgr.copy()
    for x1, y1, x2, y2, *_ in boxes:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    return out


def draw_hotspots(bgr, spots):
    out = bgr.copy()
    for s in spots:
        cv2.rectangle(out, (s["x"], s["y"]), (s["x"] + s["w"], s["y"] + s["h"]), (0, 0, 255), 3)
        cv2.putText(out, f"#{s['rank']} ({s['people']})", (s["x"], max(s["y"] - 8, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return out


# --------------------------- campus level --------------------------------
def campus_heatmap_figure(status, thresholds, grid=100, sigma=11):
    """status: DataFrame with Location, Occupancy (ratio), Count columns."""
    yy, xx = np.mgrid[0:grid, 0:grid]
    z = np.zeros((grid, grid))
    for _, r in status.iterrows():
        cfg = LOCATIONS[r["Location"]]
        z += min(r["Occupancy"], 1.2) * np.exp(-((xx - cfg["x"]) ** 2 + (yy - cfg["y"]) ** 2) / (2 * sigma**2))
    z = np.clip(z, 0, 1)
    fig = go.Figure(go.Heatmap(z=z, colorscale=PLOTLY_SCALE, zmin=0, zmax=1, showscale=True,
                               colorbar=dict(title="Congestion"), hoverinfo="skip"))
    for _, r in status.iterrows():
        cfg = LOCATIONS[r["Location"]]
        lvl = classify(r["Count"], cfg["capacity"], thresholds)
        fig.add_trace(go.Scatter(
            x=[cfg["x"]], y=[cfg["y"]], mode="markers+text",
            marker=dict(size=16, color=LEVEL_COLORS[lvl], line=dict(color="white", width=2)),
            text=[r["Location"]], textposition="top center", textfont=dict(color="white", size=12),
            hovertemplate=f"<b>{r['Location']}</b><br>{int(r['Count'])} people ({r['Occupancy']:.0%})<br>{lvl}<extra></extra>",
            showlegend=False))
    fig.update_yaxes(autorange="reversed", visible=False, scaleanchor="x")
    fig.update_xaxes(visible=False)
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0))
    return fig


def _box(cx, cy, half, height, color):
    """A solid block (Mesh3d cube) standing on the map at (cx, cy)."""
    x0, x1, y0, y1 = cx - half, cx + half, cy - half, cy + half
    return go.Mesh3d(
        x=[x0, x1, x1, x0, x0, x1, x1, x0], y=[y0, y0, y1, y1, y0, y0, y1, y1],
        z=[0, 0, 0, 0, height, height, height, height],
        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2], j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        color=color, opacity=0.95, flatshading=True, hoverinfo="skip", showlegend=False)


def campus_3d_figure(status, thresholds, grid=70, sigma=11):
    """Isometric campus: heat-coloured ground plane + a block per location
    whose height and colour follow its occupancy."""
    g = np.linspace(0, 100, grid)
    xx, yy = np.meshgrid(g, g)
    z = np.zeros_like(xx)
    for _, r in status.iterrows():
        cfg = LOCATIONS[r["Location"]]
        z += min(r["Occupancy"], 1.2) * np.exp(-((xx - cfg["x"]) ** 2 + (yy - (100 - cfg["y"])) ** 2) / (2 * sigma**2))
    z = np.clip(z, 0, 1)
    fig = go.Figure(go.Surface(x=g, y=g, z=np.zeros_like(z), surfacecolor=z, colorscale=PLOTLY_SCALE,
                               cmin=0, cmax=1, showscale=False, hoverinfo="skip", opacity=0.97,
                               lighting=dict(ambient=1, diffuse=0)))
    for _, r in status.iterrows():
        cfg = LOCATIONS[r["Location"]]
        lvl = classify(r["Count"], cfg["capacity"], thresholds)
        h = 4 + 26 * min(r["Occupancy"], 1.0)
        cx, cy = cfg["x"], 100 - cfg["y"]
        fig.add_trace(_box(cx, cy, 5.5, h, LEVEL_COLORS[lvl]))
        fig.add_trace(go.Scatter3d(
            x=[cx], y=[cy], z=[h + 7], mode="markers+text",
            marker=dict(size=5, color=LEVEL_COLORS[lvl]),
            text=[f"<b>{r['Location']}</b><br>{r['Occupancy']:.0%}"], textposition="top center",
            textfont=dict(color="white", size=11), showlegend=False,
            hovertemplate=f"<b>{r['Location']}</b><br>{int(r['Count'])} people ({r['Occupancy']:.0%})<br>{lvl}<extra></extra>"))
    axis = dict(visible=False, showbackground=False)
    fig.update_layout(
        height=470, margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(xaxis=axis, yaxis=axis, zaxis=dict(visible=False, range=[0, 60]), bgcolor="rgba(0,0,0,0)",
                   aspectmode="manual", aspectratio=dict(x=1.5, y=1.2, z=0.5),
                   camera=dict(eye=dict(x=0.85, y=-1.45, z=0.95))))
    return fig
