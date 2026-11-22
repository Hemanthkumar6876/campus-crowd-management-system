"""Campus Crowd Density & Congestion Prediction System - Streamlit dashboard.

Run:  streamlit run app.py
"""
import os
import tempfile
from datetime import datetime, timedelta

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import auth
import config as C
import database as db
import heatmap as hm
import predictor
import reports
import simulator
import style
from detector import PersonDetector, detect_video

st.set_page_config(page_title="Campus Crowd Monitor", page_icon="🏫", layout="wide")
style.inject()

# ----------------------------------------------------------------------------
# setup
# ----------------------------------------------------------------------------
db.init_db()
auth.init_users()
user = auth.require_login()   # shows the login portal and stops until signed in

if db.count_rows() == 0:
    with st.spinner("First run: generating 14 days of demo history..."):
        simulator.seed_history(days=14)

ss = st.session_state
ss.setdefault("th", C.DEFAULT_THRESHOLDS)
ss.setdefault("alert_level", "High")
ss.setdefault("last", None)


def show_df(df, **kw):
    try:
        st.dataframe(df, width="stretch", **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kw)


def show_plot(fig, **kw):
    try:
        st.plotly_chart(fig, width="stretch", theme=None, **kw)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, theme=None, **kw)


def show_img(bgr, caption=None):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    try:
        st.image(rgb, caption=caption, width="stretch")
    except TypeError:
        st.image(rgb, caption=caption, use_container_width=True)


def style_levels(df, col="Level"):
    def colour(v):
        return f"background-color:{C.LEVEL_COLORS.get(v, 'white')};color:white;font-weight:600"
    fn = getattr(df.style, "map", None) or df.style.applymap
    return fn(colour, subset=[col]).format(precision=1)


@st.cache_data(show_spinner=False)
def _load(ver):
    return db.load_observations()


def get_obs():
    return _load(db.version())


QUALITY = {"Fast": {"imgsz": 480, "size": (480, 360), "fps": 10},
           "Balanced": {"imgsz": 640, "size": (640, 480), "fps": 15},
           "Accurate": {"imgsz": 960, "size": (1280, 720), "fps": 10}}


@st.cache_resource(show_spinner="Loading detection model...")
def get_detector(backend, model_name="yolov8n.pt"):
    return PersonDetector(backend, model_name=model_name)


def latest_status(obs, at=None):
    d = obs if at is None else obs[obs.ts <= at]
    last = d.sort_values("ts").groupby("location").tail(1)
    rows = []
    for _, r in last.iterrows():
        cfg = C.LOCATIONS[r.location]
        occ = r["count"] / cfg["capacity"]
        rows.append({
            "Location": r.location, "Time": r.ts, "Count": int(r["count"]), "Capacity": cfg["capacity"],
            "Occupancy": occ, "Occupancy %": round(occ * 100, 1),
            "Density (per 100 m²)": round(100 * r["count"] / cfg["area"], 1),
            "Level": C.classify(r["count"], cfg["capacity"], ss.th),
        })
    return pd.DataFrame(rows).sort_values("Occupancy", ascending=False).reset_index(drop=True)


def maybe_alert(ts, loc, count, kind="observed"):
    cap = C.LOCATIONS[loc]["capacity"]
    lvl = C.classify(count, cap, ss.th)
    if C.level_index(lvl) >= C.level_index(ss.alert_level):
        msg = f"{lvl} congestion at {loc}: {int(count)} people ({count / cap:.0%} of capacity)"
        db.add_alert(ts, loc, lvl, count, count / cap, kind, msg)
        return msg
    return None


def badge(level):
    return (f"<span style='background:{C.LEVEL_COLORS[level]};color:white;padding:2px 10px;"
            f"border-radius:10px;font-weight:600'>{level}</span>")


# ----------------------------------------------------------------------------
# helpers for the dashboard
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _totals(ver):
    """Whole-campus head-count on a 15-minute grid."""
    o = db.load_observations()
    g = o.set_index("ts").groupby([pd.Grouper(freq="15min"), "location"])["count"].mean()
    return g.groupby(level=0).sum()


def total_at(obs_, at):
    d = obs_[(obs_.ts <= at) & (obs_.ts > at - timedelta(minutes=30))]
    if d.empty:
        return None
    return int(d.sort_values("ts").groupby("location").tail(1)["count"].sum())


def next_peak(tots, now_ts):
    """Hour of day with the highest typical campus total after the current hour."""
    wk = now_ts.dayofweek >= 5
    t = tots[(tots.index.dayofweek >= 5) == wk]
    prof = t.groupby(t.index.hour).mean()
    later = prof[prof.index > now_ts.hour]
    h = int((later if not later.empty else prof).idxmax())
    return f"{h:02d}:00 – {(h + 1) % 24:02d}:00"


def trend_figure(tots, now_ts):
    """Actual campus total today + the typical pattern continued from now (prediction)."""
    wk = now_ts.dayofweek >= 5
    today = tots[tots.index.normalize() == now_ts.normalize()]
    today = today[today.index.hour >= 6]
    if today.empty:
        today = tots.iloc[-48:]
    hist = tots[tots.index.normalize() < now_ts.normalize()]
    hist = hist[(hist.index.dayofweek >= 5) == wk]
    if hist.empty:
        hist = tots
    prof = hist.groupby(hist.index.hour * 60 + hist.index.minute).mean()
    fut = pd.date_range(today.index.max(), periods=25, freq="15min")  # next 6 h
    key = fut.hour * 60 + fut.minute
    base = prof.reindex(key).interpolate(limit_direction="both").values
    last = float(today.iloc[-1])
    ratio = last / max(base[0], 1.0)
    pred = base * (1 + (ratio - 1) * np.exp(-np.arange(len(fut)) * 0.25 / 2))  # today's deviation fades out
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=today.index, y=today.values, name="Total crowd", mode="lines",
                             line=dict(color="#38bdf8", width=3), fill="tozeroy", fillcolor="rgba(56,189,248,.10)"))
    fig.add_trace(go.Scatter(x=fut, y=pred, name="Prediction", mode="lines",
                             line=dict(color="#a78bfa", width=3, dash="dash")))
    fig.add_trace(go.Scatter(x=[today.index.max()], y=[last], mode="markers", showlegend=False,
                             marker=dict(size=11, color="white", line=dict(color="#38bdf8", width=3)),
                             hovertemplate="%{y:,.0f} people<extra></extra>"))
    fig.add_annotation(x=today.index.max(), y=last, text=f"<b>{last:,.0f}</b>", showarrow=True, arrowhead=0,
                       ay=-34, bgcolor="rgba(8,18,52,.95)", bordercolor="rgba(255,255,255,.35)", borderpad=5)
    fig.update_layout(height=250, margin=dict(l=0, r=0, t=6, b=0), yaxis_title=None, xaxis_title=None,
                      legend=dict(orientation="h", x=1, xanchor="right", y=1.18))
    fig.update_xaxes(tickformat="%H:%M")
    return fig


# ----------------------------------------------------------------------------
# sidebar
# ----------------------------------------------------------------------------
obs = get_obs()
style.sidebar_brand()
PAGES = ["Campus Overview", "Upload & Detect", "Live Camera", "Location Monitor", "Historical Database",
         "Prediction (15–30 min)", "Hotspots & Heatmaps", "Alerts", "Evaluation & Reports"]
if auth.is_admin():
    PAGES.append("User Management")
NAV = {"Campus Overview": "🏠  Dashboard", "Upload & Detect": "📷  Upload & Detect", "Live Camera": "🎥  Live Camera",
       "Location Monitor": "📍  Location Monitor", "Historical Database": "🗄️  Historical Database",
       "Prediction (15–30 min)": "📈  Prediction (15–30 min)", "Hotspots & Heatmaps": "🔥  Hotspots & Heatmaps",
       "Alerts": "🔔  Alerts", "Evaluation & Reports": "📄  Reports", "User Management": "👤  User Management"}
_s0 = latest_status(obs)
n_alerts = int((_s0["Level"].map(C.level_index) >= C.level_index(ss.alert_level)).sum())
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed",
                        format_func=lambda p: NAV[p] + (f"   ● {n_alerts}" if p == "Alerts" and n_alerts else ""))

with st.sidebar.expander("⚙️ Congestion thresholds", expanded=False):
    st.caption("Occupancy = people ÷ location capacity")
    m = st.slider("Medium starts at (%)", 5, 90, int(ss.th[0] * 100), 5)
    h = st.slider("High starts at (%)", 10, 95, int(ss.th[1] * 100), 5)
    c = st.slider("Critical starts at (%)", 15, 100, int(ss.th[2] * 100), 5)
    ss.th = tuple(sorted((m / 100, h / 100, c / 100)))
    ss.alert_level = st.selectbox("Raise an alert from level", ["Medium", "High", "Critical"],
                                  index=["Medium", "High", "Critical"].index(ss.alert_level))
st.sidebar.caption(f"{db.count_rows():,} readings in database")
style.sidebar_user(user)
if st.sidebar.button("Sign out", use_container_width=True):
    auth.logout()
style.sidebar_skyline()

# ----------------------------------------------------------------------------
# 1. Campus overview (location-wise monitoring)
# ----------------------------------------------------------------------------
if page == "Campus Overview":
    style.hero("Campus Crowd Density & Congestion Prediction System", "Monitor  •  Analyze  •  Predict  •  Manage")
    status = latest_status(obs)
    now_ts = status["Time"].max()
    tots = _totals(db.version())

    total = int(status["Count"].sum())
    before = total_at(obs, now_ts - timedelta(hours=1))
    delta = (100 * (total - before) / before) if before else 0.0
    avg_occ = total / status["Capacity"].sum()
    active = status[status["Level"].map(C.level_index) >= C.level_index(ss.alert_level)]

    k1, k2, k3, k4 = st.columns(4)
    style.kpi(k1, "blue", "👥", "Total campus crowd", f"{total:,}",
              ("up" if delta > 0.5 else "down" if delta < -0.5 else "flat", f"{abs(delta):.0f}%"))
    style.kpi(k2, "teal", style.gauge_svg(avg_occ), "Average occupancy", f"{avg_occ:.0%}")
    style.kpi(k3, "red", "⚠️", "Active alerts", f"{len(active)}", sub=f"Locations at {ss.alert_level} or above")
    style.kpi(k4, "violet", "📅", "Next peak predicted", next_peak(tots, now_ts), sub="Typical pattern for today")
    st.write("")

    map_col, hot_col = st.columns([2.1, 1])
    with map_col:
        with st.container(border=True):
            t1, t2 = st.columns([3, 1])
            with t1:
                style.panel_title("📍", "Campus view")
            view = t2.radio("View", ["3D", "2D"], horizontal=True, label_visibility="collapsed", key="map_view")
            if view == "3D":
                show_plot(hm.campus_3d_figure(status, ss.th))
            else:
                fig2 = hm.campus_heatmap_figure(status, ss.th)
                fig2.update_layout(height=470)
                show_plot(fig2)
            st.markdown("<div style='display:flex;gap:18px;color:#cbd5e1;font-size:.85rem'>" + "".join(
                f"<span><span style='display:inline-block;width:10px;height:10px;border-radius:50%;"
                f"background:{C.LEVEL_COLORS[l]};margin-right:6px'></span>{l}</span>" for l in C.LEVELS) + "</div>",
                unsafe_allow_html=True)
    with hot_col:
        with st.container(border=True):
            style.panel_title("🔥", "Campus hotspots")
            top = status.head(3)
            idx = float(min(top["Occupancy"].mean(), 1.0))
            lvl = C.classify(idx, 1, ss.th)
            st.markdown(style.donut_html(idx, C.LEVEL_COLORS[lvl], lvl, "congestion on campus"), unsafe_allow_html=True)
            old = obs[obs.ts <= now_ts - timedelta(minutes=30)]
            prev = latest_status(old).set_index("Location")["Occupancy"] if not old.empty else {}
            rows = ""
            for _, r in top.iterrows():
                p = prev.get(r.Location) if len(prev) else None
                d = 0 if p is None else r.Occupancy - p
                tr = ("<span style='color:#f87171'>▲</span>" if d > 0.03 else
                      "<span style='color:#4ade80'>▼</span>" if d < -0.03 else "<span style='color:#94a3b8'>–</span>")
                rows += (f"<div class='rank'><span class='dot' style='background:{C.LEVEL_COLORS[r.Level]}'></span>"
                         f"<span class='nm'>{r.Location}</span><span class='pc'>{r['Occupancy %']:.0f}%</span>"
                         f"<span class='tr'>{tr}</span></div>")
            st.markdown(rows, unsafe_allow_html=True)

    tr_col, al_col = st.columns([1.35, 1])
    with tr_col:
        with st.container(border=True):
            style.panel_title("📊", "Crowd trends")
            show_plot(trend_figure(tots, now_ts))
    with al_col:
        with st.container(border=True):
            style.panel_title("🔔", "Recent alerts")
            log = db.load_alerts().head(3)
            items = [(r.ts, r.location, r.level, f"{r.location}: {int(r.count)} people ({r.occupancy:.0%} of capacity)")
                     for r in log.itertuples()] if not log.empty else [
                (r.Time, r.Location, r.Level, f"{r.Location}: {r.Count} people ({r['Occupancy %']:.0f}% of capacity)")
                for _, r in active.head(3).iterrows()]
            if not items:
                st.success("No alerts. Every location is within limits.")
            html = ""
            for ts_, loc_, lv_, msg_ in items:
                col_ = C.LEVEL_COLORS[lv_]
                html += (f"<div class='arow'><span class='dot' style='background:{col_}'></span>"
                         f"<span class='tm'>{pd.Timestamp(ts_):%H:%M}</span><span class='ms'>{msg_}</span>"
                         f"<span class='pill' style='background:{col_}'>{lv_}</span></div>")
            st.markdown(html, unsafe_allow_html=True)

    with st.expander("Location details and live feed simulation"):
        cols = st.columns(4)
        for i, r in status.iterrows():
            with cols[i % 4]:
                st.markdown(
                    f"<div class='loc-card' style='border-left:5px solid {C.LEVEL_COLORS[r.Level]}'>"
                    f"<div class='name'>{r.Location}</div><div class='num'>{r.Count}</div>"
                    f"<div style='color:#94a3b8;font-size:.85rem;margin-bottom:6px'>people · {r['Occupancy %']}% of capacity</div>"
                    f"{badge(r.Level)}</div>", unsafe_allow_html=True)
        show_df(style_levels(status[["Location", "Count", "Occupancy %", "Density (per 100 m²)", "Level"]]),
                hide_index=True)
        st.markdown("**Simulate live camera feed**")
        st.caption("Adds new readings for every location (demo data) and checks alert thresholds.")
        steps = st.number_input("5-minute steps to simulate", 1, 24, 1)
        if st.button("▶ Simulate next readings"):
            rows = simulator.simulate_next(obs["ts"].max(), steps=int(steps))
            n = sum(1 for ts, loc, cnt, _ in rows if maybe_alert(ts, loc, cnt))
            st.toast(f"{len(rows)} readings added, {n} alert(s) raised")
            st.rerun()

# ----------------------------------------------------------------------------
# 2. Upload & detect  (features 1, 2, 3, 7)
# ----------------------------------------------------------------------------
elif page == "Upload & Detect":
    style.hero("Upload image / video → detect people → count → classify", "Person detection · crowd counting · density classification")
    cfg_col, out_col = st.columns([1, 2])
    with cfg_col:
        loc = st.selectbox("Camera location", list(C.LOCATIONS))
        d = st.date_input("Capture date", datetime.now().date())
        t = st.time_input("Capture time", datetime.now().time().replace(second=0, microsecond=0))
        backend = st.selectbox("Detection engine", ["auto", "yolo", "hog"],
                               help="auto = YOLOv8 if installed, otherwise OpenCV HOG")
        conf = st.slider("Detection confidence", 0.10, 0.90, 0.35, 0.05)
        factor = st.slider("Count correction factor", 0.5, 3.0, 1.0, 0.05,
                           help="Multiply raw detections to compensate for people hidden in dense crowds "
                                "(calibrate against a manual count).")
        up = st.file_uploader("Image or video", type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "mkv"])
        every, maxf = 2.0, 30
        is_video = bool(up) and up.name.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
        if is_video:
            every = st.slider("Sample a frame every (sec)", 0.5, 10.0, 2.0, 0.5)
            maxf = st.slider("Max frames to analyse", 5, 100, 30, 5)
        go_btn = st.button("🔍 Analyse", type="primary", disabled=up is None)

    if go_btn and up is not None:
        try:
            det = get_detector(backend)
            det.conf = conf
            det.imgsz, det.hog_width, det.tiles = 1280, None, False
            ts = datetime.combine(d, t)
            cap = C.LOCATIONS[loc]["capacity"]
            if is_video:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up.name)[1]) as f:
                    f.write(up.getvalue())
                    path = f.name
                bar = st.progress(0.0, text="Analysing video...")
                stats, peak_frame, all_boxes = detect_video(path, det, every, maxf, lambda p: bar.progress(min(p, 1.0)))
                bar.empty()
                os.unlink(path)
                series = pd.DataFrame(stats)
                series["adjusted"] = (series["count"] * factor).round().astype(int)
                peak_boxes = max(all_boxes, key=len)
                dmap = sum(hm.density_map(peak_frame.shape, b) for b in all_boxes)
                res = {"kind": "video", "series": series, "count_raw": int(series["count"].max()),
                       "avg": int(round(series["adjusted"].mean())), "peak": int(series["adjusted"].max())}
                base_img, boxes = peak_frame, peak_boxes
            else:
                img = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("Could not decode the image.")
                boxes = det.detect(img)
                dmap = hm.density_map(img.shape, boxes)
                base_img = img
                n = int(round(len(boxes) * factor))
                res = {"kind": "image", "count_raw": len(boxes), "avg": n, "peak": n}
            spots = hm.find_hotspots(dmap, boxes)
            res.update({"loc": loc, "ts": ts, "engine": det.backend, "boxes_img": hm.draw_boxes(base_img, boxes),
                        "heat_img": hm.draw_hotspots(hm.overlay_heatmap(base_img, dmap), spots), "spots": spots,
                        "saved": False})
            ss.last = res
            if det.error and backend == "auto":
                st.info("YOLO not available – used the OpenCV HOG detector. "
                        "Install `ultralytics` for much better accuracy in dense crowds.")
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")

    with out_col:
        r = ss.last
        if r is None:
            st.info("Upload a crowd photo or short video on the left, then press **Analyse**.")
        else:
            cap = C.LOCATIONS[r["loc"]]["capacity"]
            shown = r["peak"] if r["kind"] == "video" else r["avg"]
            lvl = C.classify(shown, cap, ss.th)
            a, b, c2, d2 = st.columns(4)
            a.metric("People counted" + (" (peak)" if r["kind"] == "video" else ""), shown)
            b.metric("Occupancy", f"{shown / cap:.0%}")
            c2.metric("Density", f"{100 * shown / C.LOCATIONS[r['loc']]['area']:.1f} /100 m²")
            d2.markdown(f"**Congestion level**<br>{badge(lvl)}", unsafe_allow_html=True)
            st.caption(f"Engine: {r['engine']} · raw detections: {r['count_raw']} · location: {r['loc']}")
            t1, t2, t3 = st.tabs(["Detections", "Heatmap & hotspots", "Details"])
            with t1:
                show_img(r["boxes_img"], "Detected people")
            with t2:
                show_img(r["heat_img"], "Density heatmap; red boxes = hotspots (ranked by people inside)")
                if r["spots"]:
                    show_df(pd.DataFrame(r["spots"])[["rank", "people", "share_pct", "x", "y", "w", "h"]]
                            .rename(columns={"share_pct": "% of crowd"}), hide_index=True)
                else:
                    st.write("No distinct hotspot found (crowd is evenly spread or too small).")
            with t3:
                if r["kind"] == "video":
                    fig = px.line(r["series"], x="time_s", y="adjusted", markers=True,
                                  labels={"time_s": "Video time (s)", "adjusted": "People"})
                    show_plot(fig)
                    show_df(r["series"], hide_index=True)
                else:
                    st.write("Single image analysed.")
            st.divider()
            mode = "peak"
            if r["kind"] == "video":
                mode = st.radio("Save to database as", ["peak", "avg"], horizontal=True,
                                format_func=lambda x: "Peak count" if x == "peak" else "Average count")
            if st.button("💾 Save to historical database", disabled=r["saved"]):
                n = r[mode]
                db.add_observations([(r["ts"], r["loc"], n, "upload")])
                msg = maybe_alert(r["ts"], r["loc"], n)
                r["saved"] = True
                st.success("Saved to history." + (f" 🚨 Alert raised: {msg}" if msg else ""))

# ----------------------------------------------------------------------------
# 2b. Live camera
# ----------------------------------------------------------------------------
elif page == "Live Camera":
    import sys
    import time
    from collections import deque

    def _backends(source, choice="Auto"):
        """Camera drivers to try, in order (None = OpenCV default)."""
        if not isinstance(source, int):
            return [None]
        if choice == "DirectShow":
            return [cv2.CAP_DSHOW]
        if choice == "Media Foundation":
            return [cv2.CAP_MSMF]
        if choice == "Default":
            return [None]
        if os.name == "nt":
            return [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]
        if sys.platform == "darwin":
            return [cv2.CAP_AVFOUNDATION, None]
        return [cv2.CAP_V4L2, None]

    def _frame_ok(f):
        """False for empty frames and for garbled 'TV static' frames (wrong pixel format / driver)."""
        if f is None or not getattr(f, "size", 0):
            return False
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
        return float(np.abs(np.diff(g.astype(np.int16), axis=1)).mean()) < 45

    def open_capture(source, choice="Auto", size=(1280, 720), warmup=True):
        """Open a webcam / stream and wait for a clean frame. For webcams it tries MJPG then the
        camera's default format, at the chosen and a safe resolution, on each driver in turn."""
        is_cam = isinstance(source, int)
        configs = list(dict.fromkeys([(True, size), (True, (640, 480)), (False, (640, 480))])) if is_cam \
            else [(False, None)]
        for be in _backends(source, choice):
            for mjpg, wh in configs:
                cap = cv2.VideoCapture(source) if be is None else cv2.VideoCapture(source, be)
                if not cap.isOpened():
                    cap.release()
                    continue
                if is_cam:
                    if mjpg:
                        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, wh[0])
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, wh[1])
                if not warmup:
                    return cap
                for _ in range(30):
                    ok, f = cap.read()
                    if ok and _frame_ok(f):
                        try:
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        except Exception:
                            pass
                        return cap
                    time.sleep(0.05)
                cap.release()
        return None

    style.hero("Live camera detection",
               "Real-time person detection, counting and alerts from a webcam, IP camera or stream")
    tab_stream = st.container()

    def _show_frame(ph, bgr):
        h, w = bgr.shape[:2]
        if w > 960:
            bgr = cv2.resize(bgr, (960, int(h * 960 / w)))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        try:
            ph.image(rgb, width="stretch")
        except TypeError:
            ph.image(rgb, use_container_width=True)

    with tab_stream:
        left_c, right_c = st.columns([1, 2])
        with left_c:
            lv_loc = st.selectbox("Camera location", list(C.LOCATIONS), key="lv_loc")
            src_type = st.radio("Video source", ["Browser camera (use this when deployed)",
                                                 "Webcam on this computer (local only)",
                                                 "IP camera / RTSP URL / video file"], key="lv_srctype")
            if src_type.startswith("Webcam"):
                source = int(st.number_input("Webcam index (0 = default camera)", 0, 10, 0, key="lv_idx"))
                cam_driver = st.selectbox("Camera driver", ["Auto", "DirectShow", "Media Foundation", "Default"],
                                          key="lv_driver", help="If the picture is garbled or black, try another driver.")
                cam_res = st.selectbox("Resolution", ["1280×720", "640×480"], key="lv_res")
            elif src_type.startswith("IP"):
                cam_driver, cam_res = "Auto", "1280×720"
                source = st.text_input("Stream URL or file path", key="lv_url",
                                       placeholder="rtsp://user:pass@192.168.1.10:554/stream  or  C:\\videos\\gate.mp4")
            else:
                cam_driver, cam_res, source = "Auto", "1280×720", None
            use_browser = src_type.startswith("Browser")
            lv_backend = st.selectbox("Detection engine", ["auto", "yolo", "hog"], key="lv_backend",
                                      help="YOLO is accurate. HOG is a basic fallback that misses many people.")
            lv_model_label = st.selectbox("YOLO model", ["yolov8n (fastest)", "yolov8s (more accurate, slower)"],
                                          key="lv_model")
            lv_model = lv_model_label.split()[0] + ".pt"
            lv_quality = st.radio("Speed / quality", list(QUALITY), index=1, horizontal=True, key="lv_quality",
                                  help="Fast = small picture, least lag. Accurate = large picture, finds smaller people.")
            lv_dense = st.checkbox("Dense crowd mode", key="lv_dense",
                                   help="Also scans 4 magnified tiles to find small / far people. About 5x slower.")
            lv_conf = st.slider("Detection confidence", 0.10, 0.90, 0.30, 0.05, key="lv_conf",
                                help="Lower finds more people but may add false ones. 0.25-0.35 is a good range for YOLO.")
            lv_factor = st.slider("Count correction factor", 0.5, 3.0, 1.0, 0.05, key="lv_factor",
                                  help="Multiplies the count, e.g. 1.2 if it usually misses 1 in 6 people.")
            if use_browser:
                every_n, max_min = 3, 10
            else:
                every_n = st.slider("Detect on every N-th frame", 1, 10, 3, key="lv_every",
                                    help="Higher = smoother video but slower reaction.")
                max_min = st.slider("Auto-stop after (minutes)", 1, 60, 10, key="lv_max")
            save_every = st.slider("Save a reading to the database every (seconds)", 5, 300, 30, 5, key="lv_save")
            qp = QUALITY[lv_quality]
            if isinstance(source, int) and st.button("🔎 Find cameras (shows a preview of each)", key="lv_scan"):
                found = 0
                prev_cols = st.columns(3)
                with st.spinner("Checking camera indexes 0-4..."):
                    for i in range(5):
                        cp = open_capture(i, cam_driver, (640, 480))
                        if cp is None:
                            continue
                        ok, fr = cp.read()
                        cp.release()
                        if ok and fr is not None:
                            prev_cols[found % 3].image(cv2.cvtColor(cv2.resize(fr, (240, 180)), cv2.COLOR_BGR2RGB),
                                                       caption=f"Index {i}")
                            found += 1
                if not found:
                    st.warning("No working camera found. Close Zoom/Teams/Meet, allow camera access in your "
                               "system privacy settings, or try another driver.")
                else:
                    st.caption("Set *Webcam index* to the one that shows your real camera picture.")
            run = False
            if use_browser:
                st.caption("Press **START** under the video and allow camera access in your browser. "
                           "Detection runs on the server, the camera stays in your browser.")
            else:
                run = st.toggle("▶ Start live detection", key="lv_run")
                st.caption("Turn the switch off to stop. The webcam / IP options are read by the machine running "
                           "the app, so a deployed app can't see your webcam: use *Browser camera* there.")
        ctx, det_b = None, None
        with right_c:
            if use_browser:
                try:
                    from streamlit_webrtc import WebRtcMode, webrtc_streamer
                    from webcam import LiveProcessor

                    det_b = get_detector(lv_backend, lv_model)
                    det_b.imgsz, det_b.hog_width, det_b.tiles = qp["imgsz"], qp["imgsz"], lv_dense
                    ice = [{"urls": ["stun:stun.l.google.com:19302"]}]
                    try:   # optional TURN server for strict networks: see README
                        t = st.secrets["turn"]
                        ice.append({"urls": [t["urls"]], "username": t["username"], "credential": t["credential"]})
                    except Exception:
                        pass
                    ctx = webrtc_streamer(
                        key=f"campus-cam-{lv_backend}-{lv_model}-{lv_quality}", mode=WebRtcMode.SENDRECV,
                        rtc_configuration={"iceServers": ice},
                        media_stream_constraints={"video": {"width": {"ideal": qp["size"][0]}, "height": {"ideal": qp["size"][1]},
                                                            "frameRate": {"ideal": qp["fps"]}}, "audio": False},
                        video_processor_factory=lambda: LiveProcessor(det_b, qp["imgsz"]), async_processing=True)
                except ImportError:
                    st.error("Browser camera needs two more packages. Run:  pip install streamlit-webrtc av")
                except Exception as exc:
                    st.error(f"Could not start the browser camera: {exc}")
            frame_ph, info_ph, status_ph = st.empty(), st.empty(), st.empty()
            if not run and not use_browser:
                frame_ph.info("Choose a source on the left and switch on **Start live detection**.")

        if use_browser and ctx is not None:
            if not ctx.state.playing:
                info_ph.info("Press **START** under the video, then click **Allow** when your browser asks for the camera.")
            else:
                cap_b = C.LOCATIONS[lv_loc]["capacity"]
                saved_b, last_b = 0, time.time()
                while ctx.state.playing:
                    p = ctx.video_processor
                    if p is not None:
                        p.every, p.factor, p.conf, p.capacity, p.th = every_n, lv_factor, lv_conf, cap_b, ss.th
                        det_b.tiles = lv_dense
                        sm = p.smooth
                        lv_b = C.classify(sm, cap_b, ss.th)
                        info_ph.markdown(
                            f"**{lv_loc}** &nbsp; {badge(lv_b)} &nbsp; **{sm}** people · {sm / cap_b:.0%} of capacity · "
                            f"{100 * sm / C.LOCATIONS[lv_loc]['area']:.1f} per 100 m² · "
                            f"engine: {det_b.backend} {det_b.model_name if det_b.backend == 'yolo' else ''} ({p.det_fps:.1f} detections/s) · readings saved: {saved_b}",
                            unsafe_allow_html=True)
                        if det_b.backend == "hog":
                            status_ph.warning("Basic detector in use (weak at finding people). Install YOLO: enable "
                                              "the ultralytics lines in requirements.txt.")
                        elif p.error:
                            status_ph.error(f"Detector error: {p.error}")
                        if p.frames > 5 and time.time() - last_b >= save_every:
                            now = datetime.now()
                            db.add_observations([(now, lv_loc, sm, "live")])
                            msg = maybe_alert(now, lv_loc, sm, "live")
                            if msg:
                                status_ph.error(f"🚨 {msg}")
                            saved_b += 1
                            last_b = time.time()
                    time.sleep(0.5)

        if run:
            if isinstance(source, str) and not source.strip():
                status_ph.error("Enter a stream URL or video file path.")
            else:
                try:
                    det = get_detector(lv_backend, lv_model)
                    det.conf = lv_conf
                    det.imgsz, det.hog_width, det.tiles = qp["imgsz"], qp["imgsz"], lv_dense
                    with st.spinner("Opening camera..."):
                        size = (1280, 720) if cam_res.startswith("1280") else (640, 480)
                        cap = open_capture(source.strip() if isinstance(source, str) else source, cam_driver, size)
                    if cap is None:
                        status_ph.error(
                            "Could not get a clean picture from the source. Try: (1) another webcam index (use *Find cameras*; "
                            "index 0 can be a virtual camera such as OBS, DroidCam or Iriun), (2) another *Camera driver*, "
                            "(3) 640×480, (4) closing apps that use the camera (Zoom, Teams, browser tabs), "
                            "(5) allowing camera access in your system privacy settings. The webcam option only works "
                            "when the app runs on the computer that has the camera.")
                    else:
                        cap_n = C.LOCATIONS[lv_loc]["capacity"]
                        hist, boxes, n, saved, fails = deque(maxlen=5), [], 0, 0, 0
                        is_file = isinstance(source, str) and not source.lower().startswith(
                            ("rtsp", "http", "rtmp", "udp", "tcp"))
                        t0, last_save, ended = time.time(), 0.0, "Stopped (time limit reached)."
                        try:
                            while time.time() - t0 < max_min * 60:
                                ok, frame = cap.read()
                                if not ok or frame is None or not frame.size:
                                    fails += 1
                                    if is_file or fails > 60:   # files end for real; cameras may just hiccup
                                        ended = "The video stream ended."
                                        break
                                    time.sleep(0.05)
                                    continue
                                fails = 0
                                if n % every_n == 0:
                                    boxes = det.detect(frame)
                                    hist.append(len(boxes) * lv_factor)
                                n += 1
                                smooth = int(round(sum(hist) / len(hist))) if hist else 0
                                lvl = C.classify(smooth, cap_n, ss.th)
                                shown = hm.draw_boxes(frame, boxes)
                                cv2.putText(shown, f"{smooth} people - {lvl}", (15, 38), cv2.FONT_HERSHEY_SIMPLEX,
                                            1.1, (255, 255, 255), 5)
                                cv2.putText(shown, f"{smooth} people - {lvl}", (15, 38), cv2.FONT_HERSHEY_SIMPLEX,
                                            1.1, (int(C.LEVEL_COLORS[lvl][5:7], 16), int(C.LEVEL_COLORS[lvl][3:5], 16),
                                                  int(C.LEVEL_COLORS[lvl][1:3], 16)), 2)
                                _show_frame(frame_ph, shown)
                                info_ph.markdown(
                                    f"**{lv_loc}** &nbsp; {badge(lvl)} &nbsp; **{smooth}** people · "
                                    f"{smooth / cap_n:.0%} of capacity · "
                                    f"{100 * smooth / C.LOCATIONS[lv_loc]['area']:.1f} per 100 m² · "
                                    f"engine: {det.backend} · readings saved: {saved}", unsafe_allow_html=True)
                                if hist and time.time() - last_save >= save_every:
                                    now = datetime.now()
                                    db.add_observations([(now, lv_loc, smooth, "live")])
                                    msg = maybe_alert(now, lv_loc, smooth, "live")
                                    if msg:
                                        status_ph.error(f"🚨 {msg}")
                                    saved += 1
                                    last_save = time.time()
                        finally:
                            cap.release()
                        status_ph.success(f"{ended} {saved} reading(s) saved to the historical database.")
                except Exception as exc:
                    status_ph.error(f"Live detection failed: {exc}")

# ----------------------------------------------------------------------------
# 3. Location monitor  (feature 4)
# ----------------------------------------------------------------------------
elif page == "Location Monitor":
    style.hero("Location-wise monitoring", "Trends, thresholds and time spent at each congestion level")
    loc = st.selectbox("Location", list(C.LOCATIONS))
    rng_opt = st.radio("Time range", ["Last 24 h", "Last 3 days", "Last 7 days", "All"], horizontal=True, index=1)
    cfg = C.LOCATIONS[loc]
    d = obs[obs.location == loc]
    if rng_opt != "All":
        hrs = {"Last 24 h": 24, "Last 3 days": 72, "Last 7 days": 168}[rng_opt]
        d = d[d.ts >= d.ts.max() - timedelta(hours=hrs)]
    if d.empty:
        st.warning("No data for this location.")
    else:
        d = d.assign(Level=[C.classify(x, cfg["capacity"], ss.th) for x in d["count"]])
        cur = d.iloc[-1]
        a, b, c2, e = st.columns(4)
        a.metric("Now", int(cur["count"]))
        b.metric("Peak in range", int(d["count"].max()))
        c2.metric("Average", f"{d['count'].mean():.0f}")
        e.markdown(f"**Current level**<br>{badge(cur.Level)}", unsafe_allow_html=True)
        fig = px.line(d, x="ts", y="count", labels={"ts": "Time", "count": "People"})
        for th_v, lv in zip(ss.th, C.LEVELS[1:]):
            fig.add_hline(y=th_v * cfg["capacity"], line_dash="dash", line_color=C.LEVEL_COLORS[lv],
                          annotation_text=f"{lv} ≥ {int(th_v * cfg['capacity'])}", annotation_position="top left")
        fig.update_layout(height=420, title=f"{loc} – crowd count (capacity {cfg['capacity']})")
        show_plot(fig)
        l, r_ = st.columns(2)
        hourly = d.assign(hour=d.ts.dt.hour).groupby("hour")["count"].mean().reset_index()
        with l:
            show_plot(px.bar(hourly, x="hour", y="count", title="Average crowd by hour of day",
                             labels={"count": "Avg people", "hour": "Hour"}))
        with r_:
            dist = d["Level"].value_counts().reindex(C.LEVELS).fillna(0).reset_index()
            dist.columns = ["Level", "Readings"]
            show_plot(px.bar(dist, x="Level", y="Readings", color="Level", color_discrete_map=C.LEVEL_COLORS,
                             title="Time spent at each congestion level"))

# ----------------------------------------------------------------------------
# 4. Historical database  (feature 5)
# ----------------------------------------------------------------------------
elif page == "Historical Database":
    style.hero("Historical crowd database", "Every reading ever captured, searchable and exportable")
    c1, c2, c3 = st.columns(3)
    locs = c1.multiselect("Locations", list(C.LOCATIONS), default=list(C.LOCATIONS))
    dmin, dmax = obs.ts.min().date(), obs.ts.max().date()
    rng = c2.date_input("Date range", (max(dmin, dmax - timedelta(days=2)), dmax), min_value=dmin, max_value=dmax)
    start, end = (rng[0], rng[-1]) if isinstance(rng, (tuple, list)) and len(rng) else (dmin, dmax)
    srcs = c3.multiselect("Source", sorted(obs.source.unique()), default=sorted(obs.source.unique()))
    f = obs[obs.location.isin(locs) & obs.source.isin(srcs) &
            (obs.ts >= pd.Timestamp(start)) & (obs.ts < pd.Timestamp(end) + timedelta(days=1))].copy()
    f["Level"] = [C.classify(n, C.LOCATIONS[l]["capacity"], ss.th) for n, l in zip(f["count"], f.location)]
    st.caption(f"{len(f):,} readings match")
    show_df(f.sort_values("ts", ascending=False).head(2000).rename(
        columns={"ts": "Time", "location": "Location", "count": "People", "source": "Source"}), hide_index=True)
    st.download_button("⬇ Download filtered history (CSV)", f.to_csv(index=False).encode(), "crowd_history.csv", "text/csv")

    with st.expander("Import readings from CSV"):
        st.caption("Columns required: `timestamp`, `location`, `count`")
        up = st.file_uploader("CSV file", type="csv", key="csvimp")
        if up and st.button("Import"):
            try:
                df = pd.read_csv(up)
                n = db.add_observations([(r.timestamp, r.location, r["count"], "import") for _, r in df.iterrows()])
                st.success(f"Imported {n} rows (unknown locations skipped).")
                st.rerun()
            except Exception as exc:
                st.error(f"Import failed: {exc}")
    if auth.is_admin():
        with st.expander("Reset database"):
            ok = st.checkbox("I understand this deletes all readings")
            if st.button("Delete all & re-generate 14-day demo history", disabled=not ok):
                db.clear_observations()
                simulator.seed_history(days=14)
                st.rerun()

# ----------------------------------------------------------------------------
# 5. Prediction  (feature 6)
# ----------------------------------------------------------------------------
elif page == "Prediction (15–30 min)":
    style.hero("Crowd prediction – next 15 and 30 minutes", "Statistical and machine-learning forecasts with early warnings")

    @st.cache_data(show_spinner="Training forecasting models...")
    def _fc(ver):
        return predictor.forecast_all(db.load_observations(), steps=6)

    @st.cache_data(show_spinner="Training machine-learning models...")
    def _fcml(ver):
        return predictor.forecast_ml(db.load_observations())

    fcs = _fc(db.version())
    mlf = _fcml(db.version())
    MODELS = ["Statistical (seasonal profile + decaying residual)", "Machine learning (Gradient Boosting)",
              "Ensemble (average of both)"]
    model = st.radio("Forecast model", MODELS, horizontal=True)

    def pick(loc, h, stat):
        ml_v = mlf.get(loc, {}).get(h, stat)
        return stat if model == MODELS[0] else ml_v if model == MODELS[1] else (stat + ml_v) / 2

    rows = []
    for loc, (raw, fc) in fcs.items():
        cap = C.LOCATIONS[loc]["capacity"]
        now_v = raw.dropna().iloc[-1]
        p15, p30 = pick(loc, 15, fc.iloc[2]["pred"]), pick(loc, 30, fc.iloc[5]["pred"])
        rows.append({"Location": loc, "Now": round(now_v), "+15 min": round(p15), "+30 min": round(p30),
                     "Level now": C.classify(now_v, cap, ss.th), "Level +15": C.classify(p15, cap, ss.th),
                     "Level +30": C.classify(p30, cap, ss.th)})
    tab = pd.DataFrame(rows)
    for _, r in tab.iterrows():
        worst = max([r["Level +15"], r["Level +30"]], key=C.level_index)
        if C.level_index(worst) >= C.level_index(ss.alert_level) and \
                C.level_index(worst) > C.level_index(r["Level now"]):
            st.warning(f"⚠️ **{r.Location}** is expected to reach **{worst}** within 30 minutes "
                       f"({r['+15 min']} → {r['+30 min']} people).")
    sel = st.selectbox("Location", list(fcs))
    raw, fc = fcs[sel]
    cap = C.LOCATIONS[sel]["capacity"]
    now_v = raw.dropna().iloc[-1]
    a, b, c2 = st.columns(3)
    a.metric("Now", f"{now_v:.0f}")
    v15, v30 = pick(sel, 15, fc.iloc[2]["pred"]), pick(sel, 30, fc.iloc[5]["pred"])
    b.metric("In 15 min", f"{v15:.0f}", f"{v15 - now_v:+.0f}")
    c2.metric("In 30 min", f"{v30:.0f}", f"{v30 - now_v:+.0f}")
    hist = raw.dropna().iloc[-72:]  # last 6 h
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Observed", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=fc["time"], y=fc["hi"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fc["time"], y=fc["lo"], fill="tonexty", line=dict(width=0),
                             fillcolor="rgba(214,40,57,.2)", name="80% interval"))
    fig.add_trace(go.Scatter(x=fc["time"], y=fc["pred"], name="Forecast", line=dict(color="#d62839", dash="dash")))
    if sel in mlf:
        fig.add_trace(go.Scatter(x=[fc["time"].iloc[2], fc["time"].iloc[5]], y=[mlf[sel][15], mlf[sel][30]],
                                 mode="markers", name="ML forecast", marker=dict(size=11, symbol="diamond", color="#f5c26b")))
    for th_v, lv in zip(ss.th, C.LEVELS[1:]):
        fig.add_hline(y=th_v * cap, line_dash="dot", line_color=C.LEVEL_COLORS[lv], annotation_text=lv)
    fig.update_layout(height=430, title=f"{sel}: last 6 hours and next 30 minutes", yaxis_title="People")
    show_plot(fig)
    st.subheader("Campus-wide forecast")
    sty_fn = getattr(tab.style, "map", None) or tab.style.applymap
    show_df(sty_fn(lambda v: f"background-color:{C.LEVEL_COLORS.get(v, 'white')};color:white;font-weight:600",
                   subset=["Level now", "Level +15", "Level +30"]), hide_index=True)
    st.caption("Statistical model: typical time-of-day profile + today's deviation that decays over time. "
               "ML model: scikit-learn gradient boosting on time-of-day, recent counts and the typical profile. "
               "See *Evaluation & Reports* for measured accuracy.")

# ----------------------------------------------------------------------------
# 6. Hotspots & heatmaps  (feature 7)
# ----------------------------------------------------------------------------
elif page == "Hotspots & Heatmaps":
    style.hero("Hotspots & heatmaps", "Where and when the campus gets crowded")
    t1, t2, t3 = st.tabs(["Campus heatmap at a chosen time", "Hotspot ranking & busy hours", "Detected hotspots (last upload)"])
    with t1:
        c1, c2 = st.columns(2)
        day = c1.date_input("Day", obs.ts.max().date(), min_value=obs.ts.min().date(), max_value=obs.ts.max().date())
        tm = c2.time_input("Time", datetime.now().time().replace(second=0, microsecond=0))
        at = datetime.combine(day, tm)
        sub = obs[(obs.ts <= at) & (obs.ts >= at - timedelta(minutes=30))]
        if sub.empty:
            st.warning("No readings within 30 minutes before that time.")
        else:
            stt = latest_status(sub)
            show_plot(hm.campus_heatmap_figure(stt, ss.th))
            show_df(style_levels(stt[["Location", "Count", "Occupancy %", "Level"]]), hide_index=True)
    with t2:
        recent = obs[obs.ts >= obs.ts.max() - timedelta(days=7)].copy()
        recent["occ"] = recent["count"] / recent.location.map(lambda l: C.LOCATIONS[l]["capacity"])
        recent["lvl"] = [C.level_index_array([n], [C.LOCATIONS[l]["capacity"]], ss.th)[0]
                         for n, l in zip(recent["count"], recent.location)]
        rank = recent.groupby("location").agg(avg_occ=("occ", "mean"), peak_occ=("occ", "max"),
                                              pct_high=("lvl", lambda s: 100 * (s >= 2).mean())).reset_index()
        rank.columns = ["Location", "Avg occupancy", "Peak occupancy", "% time High/Critical"]
        rank = rank.sort_values("% time High/Critical", ascending=False).round(2)
        st.subheader("Chronic hotspots (last 7 days)")
        show_df(rank, hide_index=True)
        wk = st.radio("Days", ["Weekdays", "Weekends", "All"], horizontal=True)
        r2 = recent
        if wk == "Weekdays":
            r2 = r2[r2.ts.dt.dayofweek < 5]
        elif wk == "Weekends":
            r2 = r2[r2.ts.dt.dayofweek >= 5]
        piv = r2.assign(hour=r2.ts.dt.hour).pivot_table(index="location", columns="hour", values="occ", aggfunc="mean")
        fig = px.imshow(piv * 100, aspect="auto", color_continuous_scale=hm.PLOTLY_SCALE, zmin=0, zmax=100,
                        labels=dict(x="Hour of day", y="", color="Occupancy %"),
                        title="Average occupancy % by location and hour")
        show_plot(fig)
    with t3:
        r = ss.last
        if r is None:
            st.info("Analyse an image or video on the *Upload & Detect* page first.")
        else:
            show_img(r["heat_img"], f"{r['loc']} – {r['ts']:%d %b %H:%M}")
            if r["spots"]:
                show_df(pd.DataFrame(r["spots"]), hide_index=True)

# ----------------------------------------------------------------------------
# 7. Alerts  (feature 8)
# ----------------------------------------------------------------------------
elif page == "Alerts":
    style.hero("Threshold-based congestion alerts", "Active, predicted and logged alerts")
    st.caption(f"Alerts are raised from **{ss.alert_level}** upward. "
               "Change thresholds in the sidebar (⚙️ Congestion thresholds).")
    status = latest_status(obs)
    active = status[status["Level"].map(C.level_index) >= C.level_index(ss.alert_level)]
    st.subheader("Active now")
    if active.empty:
        st.success("✅ No location is currently above the alert threshold.")
    for _, r in active.iterrows():
        st.error(f"**{r.Level}** – {r.Location}: {r.Count} people ({r['Occupancy %']}% of capacity)")

    st.subheader("Predicted within 30 minutes")
    fcs = predictor.forecast_all(obs, steps=6)
    any_pred = False
    for loc, (raw, fc) in fcs.items():
        cap = C.LOCATIONS[loc]["capacity"]
        lv = [C.classify(v, cap, ss.th) for v in fc["pred"]]
        worst = max(lv, key=C.level_index)
        if C.level_index(worst) >= C.level_index(ss.alert_level):
            any_pred = True
            when = fc["time"].iloc[[C.level_index(x) >= C.level_index(ss.alert_level) for x in lv].index(True)]
            st.warning(f"⚠️ {loc}: forecast **{worst}** (first above threshold at {when:%H:%M}).")
    if not any_pred:
        st.success("No congestion expected in the next 30 minutes.")

    st.subheader("Alert log")
    log = db.load_alerts()
    if log.empty:
        st.info("No alerts have been logged yet. Save an upload or simulate live readings to generate some.")
    else:
        show_only = st.checkbox("Only unacknowledged", True)
        view = log[log.acknowledged == 0] if show_only else log
        show_df(style_levels(view[["id", "ts", "location", "level", "count", "kind", "message", "acknowledged"]], "level"),
                hide_index=True)
        c1, c2, c3 = st.columns(3)
        if c1.button("✔ Acknowledge all"):
            db.acknowledge_alerts()
            st.rerun()
        if auth.is_admin():
            if c2.button("🗑 Clear log"):
                db.clear_alerts()
                st.rerun()
        c3.download_button("⬇ Download alert log (CSV)", log.to_csv(index=False).encode(), "alerts.csv", "text/csv")

# ----------------------------------------------------------------------------
# 8. Evaluation & reports  (feature 10)
# ----------------------------------------------------------------------------
elif page == "Evaluation & Reports":
    style.hero("Prediction accuracy & reports", "Walk-forward back-test and downloadable reports")
    c1, c2 = st.columns(2)
    test_days = c1.slider("Back-test period (last N days)", 1, 7, 3)
    st.caption("Walk-forward test: the model learns only from data *before* the test window, then predicts every "
               "5-minute origin inside it. Persistence ('same as now') is the baseline to beat.")

    @st.cache_data(show_spinner="Running back-test...")
    def _eval(ver, th, days):
        return predictor.evaluate(db.load_observations(), th, days)

    metrics, details = _eval(db.version(), ss.th, test_days)
    if metrics.empty:
        st.warning("Not enough history for evaluation.")
    else:
        allc = metrics[metrics.Location == "ALL CAMPUS"]
        cols = st.columns(2 * len(allc))
        for i, (_, r) in enumerate(allc.iterrows()):
            cols[2 * i].metric(f"{int(r.Horizon)}-min MAE · statistical", f"{r['MAE (people)']:.1f} people",
                               f"{r['Improvement vs baseline %']:+.0f}% vs persistence")
            cols[2 * i].caption(f"Level accuracy {r['Level accuracy %']:.0f}% · WAPE {r['WAPE %']:.0f}%")
            cols[2 * i + 1].metric(f"{int(r.Horizon)}-min MAE · ML", f"{r['ML MAE (people)']:.1f} people",
                                   f"{r['ML Improvement vs baseline %']:+.0f}% vs persistence")
            cols[2 * i + 1].caption(f"Level accuracy {r['ML Level accuracy %']:.0f}%")
        st.subheader("Metrics by location")
        show_df(metrics, hide_index=True)
        l, r_ = st.columns(2)
        per = metrics[metrics.Location != "ALL CAMPUS"].copy()
        per["Horizon"] = per["Horizon"].astype(int).astype(str) + " min"
        with l:
            cmp_ = per.melt(id_vars=["Location", "Horizon"], value_vars=["MAE (people)", "ML MAE (people)"],
                            var_name="Model", value_name="MAE")
            cmp_["Model"] = cmp_["Model"].map({"MAE (people)": "Statistical", "ML MAE (people)": "ML"}) + " " + cmp_["Horizon"]
            show_plot(px.bar(cmp_, x="Location", y="MAE", color="Model", barmode="group",
                             title="Mean absolute error by location and model"))
        with r_:
            show_plot(px.bar(per, x="Location", y="Level accuracy %", color="Horizon", barmode="group",
                             title="Congestion-level accuracy"))
        st.subheader("Predicted vs actual")
        s1, s2 = st.columns(2)
        sel = s1.selectbox("Location", list(C.LOCATIONS))
        hor = s2.selectbox("Horizon (min)", sorted(details.Horizon.unique()))
        dd = details[(details.Location == sel) & (details.Horizon == hor)].sort_values("time")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dd.time, y=dd.actual, name="Actual"))
        fig.add_trace(go.Scatter(x=dd.time, y=dd.pred, name="Statistical", line=dict(dash="dash")))
        fig.add_trace(go.Scatter(x=dd.time, y=dd.pred_ml, name="ML (gradient boosting)", line=dict(dash="dash")))
        fig.add_trace(go.Scatter(x=dd.time, y=dd.baseline, name="Persistence", line=dict(dash="dot", color="gray")))
        fig.update_layout(height=400, yaxis_title="People")
        show_plot(fig)
        g = details[details.Horizon == hor]
        la = C.level_index_array(g.actual, g.capacity, ss.th)
        lp = C.level_index_array(g.pred, g.capacity, ss.th)
        cm = pd.crosstab(pd.Series(la, name="Actual"), pd.Series(lp, name="Predicted")).reindex(
            index=range(4), columns=range(4), fill_value=0)
        cm.index = cm.columns = C.LEVELS
        show_plot(px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                            title=f"Level confusion matrix ({hor}-min, all locations)",
                            labels=dict(x="Predicted", y="Actual")))

    st.subheader("Downloads")
    status = latest_status(obs)
    d1, d2, d3, d4 = st.columns(4)
    try:
        pdf = reports.build_pdf(status, db.load_alerts(), metrics if not metrics.empty else None,
                                details if not metrics.empty else None, ss.th)
        d1.download_button("📄 Full PDF report", pdf, f"campus_crowd_report_{datetime.now():%Y%m%d_%H%M}.pdf",
                           "application/pdf")
    except Exception as exc:
        d1.error(f"PDF failed: {exc}")
    d2.download_button("⬇ Current status (CSV)", status.drop(columns=["Occupancy"]).to_csv(index=False).encode(),
                       "current_status.csv", "text/csv")
    if not metrics.empty:
        d3.download_button("⬇ Accuracy metrics (CSV)", metrics.to_csv(index=False).encode(), "accuracy_metrics.csv", "text/csv")
        d4.download_button("⬇ Predictions vs actuals (CSV)", details.to_csv(index=False).encode(),
                           "predictions_vs_actuals.csv", "text/csv")

# ----------------------------------------------------------------------------
# 9. User management (admin only)
# ----------------------------------------------------------------------------
elif page == "User Management":
    if auth.is_admin():
        auth.user_management_page()
    else:
        st.error("Only administrators can manage users.")
