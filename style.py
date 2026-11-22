"""Visual theme: deep-navy command-centre look with neon-blue glass panels.

Also holds the small HTML components used by the dashboard (top bar, KPI cards,
gauge, donut, sidebar brand, login brand panel).
"""
from datetime import datetime

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

COLORWAY = ["#38bdf8", "#a78bfa", "#34d399", "#fbbf24", "#fb7185", "#94a3b8"]

_tpl = go.layout.Template(pio.templates["plotly_dark"])
_tpl.layout.paper_bgcolor = "rgba(0,0,0,0)"
_tpl.layout.plot_bgcolor = "rgba(0,0,0,0)"
_tpl.layout.colorway = COLORWAY
_tpl.layout.font = dict(family="Inter, sans-serif", color="#cdd9f5")
_tpl.layout.xaxis.gridcolor = "rgba(120,160,255,.10)"
_tpl.layout.yaxis.gridcolor = "rgba(120,160,255,.10)"
pio.templates["campus"] = _tpl
pio.templates.default = "campus"

# a simple line-art campus building used as the logo
LOGO = ("<svg viewBox='0 0 64 64' width='{s}' height='{s}' fill='none' stroke='#5cc8ff' stroke-width='2.4' "
        "stroke-linecap='round' stroke-linejoin='round'>"
        "<path d='M8 56h48M12 56V30h40v26M32 8l22 14H10L32 8z'/>"
        "<path d='M20 56V36M28 56V36M36 56V36M44 56V36'/><circle cx='32' cy='20' r='2.6'/></svg>")

SKYLINE = ("<svg viewBox='0 0 240 70' width='100%' preserveAspectRatio='xMidYMax slice' fill='none' "
           "stroke='#4aa8ff' stroke-width='1.3' opacity='.55'>"
           "<path d='M0 68h240M10 68V40h30v28M40 68V30h24v38M64 68V44h28v28M92 68V22h8V10h14v12h8v46"
           "M122 68V36h30v32M152 68V28h26v40M178 68V42h30v26M208 68V34h26v34'/>"
           "<path d='M16 48h18M16 56h18M46 38h12M46 46h12M46 54h12M72 52h12M130 44h14M130 52h14M158 36h14M158 44h14'/></svg>")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Poppins:wght@500;600;700&display=swap');
:root{
  --bg0:#040916; --bg1:#081231; --panel:rgba(16,30,78,.52); --panel2:rgba(20,38,96,.40);
  --line:rgba(96,150,255,.28); --line2:rgba(96,150,255,.14);
  --cyan:#38bdf8; --violet:#8b5cf6; --txt:#e8eefc; --mut:#8fa3cc;
  --green:#22c55e; --amber:#f59e0b; --orange:#f97316; --red:#ef4444;
}
html, body, [class*="css"], .stApp { font-family:'Inter',sans-serif; color:var(--txt); }
h1,h2,h3,h4 { font-family:'Poppins',sans-serif !important; letter-spacing:.1px; }
.stApp {
  background:
    radial-gradient(900px 520px at 12% -10%, rgba(56,120,255,.30), transparent 62%),
    radial-gradient(800px 500px at 100% 0%, rgba(139,92,246,.24), transparent 60%),
    radial-gradient(900px 500px at 55% 125%, rgba(30,110,255,.16), transparent 60%),
    linear-gradient(165deg, var(--bg0) 0%, var(--bg1) 55%, #0a1640 100%);
  background-attachment: fixed;
}
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top:1.2rem; max-width:1500px; }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {
  background: linear-gradient(185deg, rgba(8,18,52,.97), rgba(5,11,34,.98));
  border-right:1px solid var(--line2);
}
section[data-testid="stSidebar"] > div { padding-top:.4rem; }
.brand { text-align:center; padding:14px 4px 10px; }
.brand .t { font-family:'Poppins',sans-serif; font-weight:600; font-size:1.18rem; line-height:1.2; margin-top:6px; color:#fff; }
section[data-testid="stSidebar"] div[role="radiogroup"] { gap:6px; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label {
  width:100%; padding:11px 14px; border-radius:12px; border:1px solid transparent;
  color:#b6c5e8; cursor:pointer; transition: background .15s, border-color .15s;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child { display:none; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label p { font-size:.92rem; font-weight:500; margin:0; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover { background:rgba(80,140,255,.10); }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
  background: linear-gradient(135deg, #1d6bff, #2d8bff 60%, #3aa0ff); color:#fff;
  border-color:rgba(140,190,255,.6); box-shadow:0 0 22px rgba(45,130,255,.45);
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p { color:#fff; font-weight:600; }
.usercard { display:flex; align-items:center; gap:10px; padding:10px 12px; margin:10px 0 6px;
  border-radius:14px; background:var(--panel2); border:1px solid var(--line2); }
.usercard .nm { font-weight:600; font-size:.9rem; line-height:1.15; }
.usercard .rl { font-size:.72rem; color:var(--mut); }
.skyline { margin-top:14px; border-radius:12px; overflow:hidden; opacity:.9;
  background:linear-gradient(180deg, transparent, rgba(40,110,255,.18)); }
.avatar { width:38px; height:38px; border-radius:50%; flex:none; display:flex; align-items:center; justify-content:center;
  font-weight:700; color:#fff; background:linear-gradient(135deg,#38bdf8,#6366f1); border:2px solid rgba(255,255,255,.25); }

/* ---------- top bar ---------- */
.topbar { display:flex; align-items:center; gap:18px; padding:16px 24px; margin-bottom:16px; border-radius:22px;
  background:linear-gradient(120deg, rgba(30,70,170,.42), rgba(18,34,90,.55) 55%, rgba(60,40,140,.38));
  border:1px solid var(--line); box-shadow:0 10px 40px rgba(0,0,0,.40), inset 0 0 40px rgba(60,120,255,.08); }
.topbar .ttl { flex:1; min-width:0; }
.topbar h1 { font-size:1.7rem; font-weight:700; margin:0; color:#fff; line-height:1.15; }
.topbar p { margin:4px 0 0; color:#b9c8ee; font-size:.9rem; }
.topbar .right { display:flex; align-items:center; gap:16px; white-space:nowrap; }
.live { display:inline-flex; align-items:center; gap:8px; padding:6px 16px; border-radius:999px; font-weight:600;
  background:rgba(34,197,94,.16); border:1px solid rgba(34,197,94,.55); color:#eafff2; }
.live i { width:9px; height:9px; border-radius:50%; background:var(--green); box-shadow:0 0 10px var(--green); }
.clock { color:#d6e1ff; font-size:.95rem; }

/* ---------- KPI cards ---------- */
.kpi { display:flex; align-items:center; gap:16px; padding:16px 20px; border-radius:20px; height:104px;
  border:1px solid var(--line); box-shadow:0 8px 28px rgba(0,0,0,.32); }
.kpi.blue  { background:linear-gradient(135deg, rgba(30,90,220,.40), rgba(16,32,86,.55)); }
.kpi.teal  { background:linear-gradient(135deg, rgba(20,150,170,.34), rgba(16,32,86,.55)); border-color:rgba(60,200,220,.40); }
.kpi.red   { background:linear-gradient(135deg, rgba(190,40,90,.38), rgba(40,20,70,.55)); border-color:rgba(240,90,130,.40); }
.kpi.violet{ background:linear-gradient(135deg, rgba(110,70,220,.40), rgba(24,24,80,.55)); border-color:rgba(160,120,255,.40); }
.kpi .ico { width:58px; height:58px; flex:none; display:flex; align-items:center; justify-content:center; font-size:2.1rem; }
.kpi .lb { color:#dbe6ff; font-size:.85rem; }
.kpi .vl { font-family:'Poppins',sans-serif; font-weight:600; font-size:1.95rem; line-height:1.15; color:#fff; }
.kpi .vl small { font-size:.8rem; font-weight:600; margin-left:6px; }
.kpi .up { color:#4ade80; } .kpi .down { color:#fb923c; } .kpi .flat { color:#94a3b8; }
.kpi .sub { color:#ffb4c4; font-size:.82rem; }

/* ---------- panels (st.container(border=True)) ---------- */
div[data-testid="stVerticalBlockBorderWrapper"] {
  background:var(--panel); border:1px solid var(--line) !important; border-radius:20px !important;
  box-shadow:0 8px 30px rgba(0,0,0,.30), inset 0 0 36px rgba(60,120,255,.05); backdrop-filter: blur(10px);
}
.ptitle { display:flex; align-items:center; gap:10px; font-family:'Poppins',sans-serif; font-weight:600; font-size:1.08rem; color:#fff; margin:2px 0 6px; }
.rank { display:flex; align-items:center; gap:12px; padding:11px 4px; border-top:1px solid var(--line2); font-size:.98rem; }
.rank .dot { width:13px; height:13px; border-radius:50%; flex:none; }
.rank .nm { flex:1; } .rank .pc { font-weight:600; width:48px; text-align:right; }
.rank .tr { width:20px; text-align:center; font-size:.9rem; }
.arow { display:flex; align-items:center; gap:12px; padding:11px 2px; border-top:1px solid var(--line2); font-size:.93rem; }
.arow .dot { width:11px; height:11px; border-radius:50%; flex:none; }
.arow .tm { color:#a9b9e0; width:44px; } .arow .ms { flex:1; }
.pill { display:inline-block; padding:3px 14px; border-radius:999px; font-weight:600; font-size:.8rem; color:#fff; }

/* ---------- metrics (other pages) ---------- */
div[data-testid="stMetric"] { background:var(--panel2); border:1px solid var(--line2); padding:14px 18px; border-radius:16px; }
div[data-testid="stMetricLabel"] p { color:var(--mut); font-size:.78rem; }
div[data-testid="stMetricValue"] { color:#fff; font-weight:600; font-size:1.5rem; }
div[data-testid="stMetricValue"] > div { white-space:normal !important; overflow:visible !important; text-overflow:clip !important; line-height:1.2; }
.loc-card { padding:12px 16px; margin-bottom:12px; border-radius:16px; background:var(--panel2); border:1px solid var(--line2); }
.loc-card .name { color:#cbd5e1; font-weight:500; font-size:.9rem; }
.loc-card .num { font-size:30px; font-weight:700; color:#fff; line-height:1.15; }

/* ---------- controls ---------- */
button[data-baseweb="tab"] { font-weight:500; }
button[data-baseweb="tab"][aria-selected="true"] { color:#5cc8ff; }
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
  border-radius:12px; border:1px solid var(--line); color:#fff; font-weight:500;
  background:linear-gradient(135deg, rgba(60,120,255,.22), rgba(56,189,248,.12)); transition:all .15s ease; }
.stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {
  border-color:#5cc8ff; box-shadow:0 0 18px rgba(56,189,248,.35); }
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"],
.stFormSubmitButton > button[data-testid="stBaseButton-primary"], .stButton > button[data-testid="stBaseButton-primary"] {
  background:linear-gradient(135deg,#1d6bff,#3aa0ff); border:none; box-shadow:0 6px 20px rgba(40,120,255,.4); }
div[data-testid="stDataFrame"], div[data-testid="stPlotlyChart"] { border-radius:14px; }
div[data-testid="stFileUploader"] section { background:rgba(255,255,255,.03); border:1px dashed rgba(96,150,255,.5); border-radius:14px; }
hr { border-color:var(--line2); }
h2, h3 { color:#f1f5f9; }
:focus-visible { outline:2px solid #5cc8ff !important; outline-offset:2px; }

/* 3D / 2D switch: horizontal radio in the main area rendered as pills */
section[data-testid="stMain"] div[role="radiogroup"][aria-orientation="horizontal"] > label {
  padding:4px 16px; border-radius:999px; border:1px solid var(--line2); background:rgba(255,255,255,.03); }
section[data-testid="stMain"] div[role="radiogroup"][aria-orientation="horizontal"] > label > div:first-child { display:none; }
section[data-testid="stMain"] div[role="radiogroup"][aria-orientation="horizontal"] > label:has(input:checked) {
  background:linear-gradient(135deg,#1d6bff,#3aa0ff); border-color:transparent; }
@media (max-width: 900px){ .topbar{flex-wrap:wrap} .topbar .right{width:100%} }
</style>
"""

LOGIN_CSS = """
<style>
section[data-testid="stSidebar"], div[data-testid="stSidebarCollapsedControl"], header[data-testid="stHeader"] { display:none !important; }
.block-container { padding-top:2.2rem; max-width:1180px; }
.login-brand { min-height:640px; padding:44px 40px 0; border-radius:26px; position:relative; overflow:hidden;
  background:
    radial-gradient(600px 360px at 20% 0%, rgba(56,130,255,.45), transparent 62%),
    radial-gradient(500px 380px at 100% 90%, rgba(139,92,246,.35), transparent 60%),
    linear-gradient(160deg, rgba(14,30,90,.85), rgba(6,14,44,.95));
  border:1px solid var(--line); box-shadow:0 20px 60px rgba(0,0,0,.5); }
.login-brand h1 { font-size:2.5rem; line-height:1.12; margin:18px 0 12px; color:#fff; font-weight:700; }
.login-brand p.lead { color:#b9c8ee; font-size:1.02rem; max-width:30rem; line-height:1.55; }
.login-brand ul { list-style:none; padding:0; margin:26px 0 0; display:grid; gap:12px; }
.login-brand li { display:flex; gap:12px; align-items:center; color:#dbe6ff; font-size:.97rem; }
.login-brand li b { width:34px; height:34px; border-radius:10px; display:flex; align-items:center; justify-content:center;
  background:rgba(60,130,255,.22); border:1px solid var(--line); font-size:1rem; }
.login-brand .sky { position:absolute; left:0; right:0; bottom:0; height:150px; }
.login-spacer { height:60px; }
.login-title { font-family:'Poppins',sans-serif; font-size:1.7rem; font-weight:600; color:#fff; margin-top:6px; }
.login-sub { color:var(--mut); margin:2px 0 12px; }
@media (max-width: 900px){ .login-brand{ min-height:auto; padding-bottom:170px } .login-spacer{height:0} }
</style>
"""


def inject():
    st.markdown(CSS, unsafe_allow_html=True)


def login_css():
    st.markdown(LOGIN_CSS, unsafe_allow_html=True)


def login_brand_html():
    return (f"<div class='login-brand'>{LOGO.format(s=64)}"
            "<h1>Campus Crowd Density &amp; Congestion Prediction</h1>"
            "<p class='lead'>See how busy every part of campus is right now, and where it will be busy in the next 30 minutes.</p>"
            "<ul><li><b>👥</b>Live people counts from camera images and video</li>"
            "<li><b>🔥</b>Heatmaps that show where crowds are building</li>"
            "<li><b>📈</b>15 to 30 minute congestion forecasts</li>"
            "<li><b>🔔</b>Alerts when a location gets too crowded</li></ul>"
            f"<div class='sky'>{SKYLINE}</div></div>")


# ------------------------------------------------------------------ components
def _initials(name):
    parts = [p for p in str(name).split() if p]
    return ("".join(p[0] for p in parts[:2]) or "?").upper()


def hero(title, subtitle=""):
    """Top bar used on every page: logo, title, live pill, clock, avatar."""
    u = st.session_state.get("user") or {"name": "Guest"}
    now = datetime.now()
    st.markdown(
        f"<div class='topbar'>{LOGO.format(s=52)}"
        f"<div class='ttl'><h1>{title}</h1><p>{subtitle}</p></div>"
        f"<div class='right'><span class='live'><i></i>Live</span>"
        f"<span class='clock'>{now:%a, %b %d, %Y}&nbsp;&nbsp;|&nbsp;&nbsp;{now:%H:%M}</span>"
        f"<div class='avatar' title='{u['name']}'>{_initials(u['name'])}</div></div></div>",
        unsafe_allow_html=True)


def sidebar_brand():
    st.sidebar.markdown(f"<div class='brand'>{LOGO.format(s=58)}<div class='t'>Campus<br>Crowd Monitor</div></div>",
                        unsafe_allow_html=True)


def sidebar_user(user):
    st.sidebar.markdown(
        f"<div class='usercard'><div class='avatar'>{_initials(user['name'])}</div>"
        f"<div><div class='nm'>{user['name']}</div><div class='rl'>{user['role'].title()}</div></div></div>",
        unsafe_allow_html=True)


def sidebar_skyline():
    st.sidebar.markdown(f"<div class='skyline'>{SKYLINE}</div>", unsafe_allow_html=True)


def kpi(col, tone, icon, label, value, extra=None, sub=""):
    """icon: emoji or raw <svg> html. extra: ('up'|'down'|'flat', '12%')."""
    ex = ""
    if extra:
        cls, txt = extra
        arrow = {"up": "↑", "down": "↓", "flat": "→"}[cls]
        ex = f"<small class='{cls}'>{arrow} {txt}</small>"
    sub_html = f"<div class='sub'>{sub}</div>" if sub else ""
    col.markdown(f"<div class='kpi {tone}'><div class='ico'>{icon}</div><div>"
                 f"<div class='lb'>{label}</div><div class='vl'>{value}{ex}</div>{sub_html}</div></div>",
                 unsafe_allow_html=True)


def gauge_svg(pct):
    """Half-circle gauge, pct 0..1."""
    pct = max(0.0, min(1.0, pct))
    arc = 125.66
    ang = -90 + 180 * pct
    return ("<svg viewBox='0 0 100 62' width='84' height='52'>"
            "<defs><linearGradient id='g' x1='0' x2='1'><stop offset='0' stop-color='#22c55e'/>"
            "<stop offset='.55' stop-color='#facc15'/><stop offset='1' stop-color='#ef4444'/></linearGradient></defs>"
            "<path d='M10 52A40 40 0 0 1 90 52' fill='none' stroke='rgba(255,255,255,.14)' stroke-width='9' stroke-linecap='round'/>"
            f"<path d='M10 52A40 40 0 0 1 90 52' fill='none' stroke='url(#g)' stroke-width='9' stroke-linecap='round' "
            f"stroke-dasharray='{arc * pct:.1f} {arc}'/>"
            f"<g transform='rotate({ang:.1f} 50 52)'><path d='M50 52L50 22' stroke='#e8eefc' stroke-width='3' stroke-linecap='round'/></g>"
            "<circle cx='50' cy='52' r='5' fill='#e8eefc'/></svg>")


def donut_html(pct, color, label_top, label_bottom):
    pct = max(0.0, min(1.0, pct))
    c = 2 * 3.14159 * 42
    return (
        "<div style='display:flex;align-items:center;gap:18px;margin:6px 0 4px'>"
        "<svg viewBox='0 0 100 100' width='150' height='150' style='flex:none'>"
        "<circle cx='50' cy='50' r='42' fill='none' stroke='rgba(255,255,255,.10)' stroke-width='11'/>"
        f"<circle cx='50' cy='50' r='42' fill='none' stroke='{color}' stroke-width='11' stroke-linecap='round' "
        f"stroke-dasharray='{c * pct:.1f} {c:.1f}' transform='rotate(-90 50 50)' style='filter:drop-shadow(0 0 6px {color})'/>"
        f"<text x='50' y='57' text-anchor='middle' font-family='Poppins,sans-serif' font-weight='700' font-size='23' fill='#fff'>{pct:.0%}</text></svg>"
        f"<div style='font-family:Poppins,sans-serif;font-weight:600;line-height:1.25'>"
        f"<div style='color:{color};font-size:1.25rem'>{label_top}</div>"
        f"<div style='color:#fff;font-size:1.05rem'>{label_bottom}</div></div></div>")


def panel_title(icon, text):
    st.markdown(f"<div class='ptitle'><span>{icon}</span>{text}</div>", unsafe_allow_html=True)
