"""Visual theme: dark navy/gold 'classy' look with glass cards and themed Plotly charts."""
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

COLORWAY = ["#38bdf8", "#f5c26b", "#a78bfa", "#34d399", "#fb7185", "#94a3b8"]

_tpl = go.layout.Template(pio.templates["plotly_dark"])
_tpl.layout.paper_bgcolor = "rgba(0,0,0,0)"
_tpl.layout.plot_bgcolor = "rgba(0,0,0,0)"
_tpl.layout.colorway = COLORWAY
_tpl.layout.font = dict(family="Inter, sans-serif", color="#dbe4f3")
_tpl.layout.xaxis.gridcolor = "rgba(148,163,184,.12)"
_tpl.layout.yaxis.gridcolor = "rgba(148,163,184,.12)"
pio.templates["campus"] = _tpl
pio.templates.default = "campus"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@500;600;700&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }

.stApp {
  background:
    radial-gradient(1100px 600px at 8% -8%, rgba(56,189,248,.20), transparent 60%),
    radial-gradient(900px 520px at 100% 0%, rgba(167,139,250,.18), transparent 60%),
    radial-gradient(900px 600px at 50% 120%, rgba(245,194,107,.10), transparent 60%),
    linear-gradient(160deg, #080d1a 0%, #0d1428 45%, #101a33 100%);
  background-attachment: fixed;
}
/* faint architectural grid */
.stApp::before {
  content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0;
  background-image:
    linear-gradient(rgba(148,163,184,.045) 1px, transparent 1px),
    linear-gradient(90deg, rgba(148,163,184,.045) 1px, transparent 1px);
  background-size: 56px 56px;
  mask-image: radial-gradient(ellipse at center, black 30%, transparent 85%);
}
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; max-width: 1400px; position: relative; z-index: 1; }

/* sidebar */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(15,23,42,.92), rgba(10,15,30,.96));
  border-right: 1px solid rgba(245,194,107,.18);
}
section[data-testid="stSidebar"] h1 {
  font-family: 'Playfair Display', serif; font-size: 1.55rem; color: #f5c26b; letter-spacing:.3px;
}

/* hero */
.hero {
  padding: 26px 30px; margin-bottom: 22px; border-radius: 18px;
  background: linear-gradient(120deg, rgba(56,189,248,.14), rgba(167,139,250,.10) 55%, rgba(245,194,107,.12));
  border: 1px solid rgba(255,255,255,.08); box-shadow: 0 10px 40px rgba(0,0,0,.35);
  backdrop-filter: blur(10px);
}
.hero h1 { font-family:'Playfair Display', serif; font-weight:600; font-size:2.2rem; margin:0; color:#fff; }
.hero p { margin:6px 0 0; color:#a9b6cf; font-size:.98rem; }
.hero .gold { height:3px; width:70px; border-radius:2px; margin-top:14px;
  background: linear-gradient(90deg,#f5c26b,#fde3a7); }

/* glass cards */
div[data-testid="stMetric"] {
  background: rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08);
  padding: 14px 18px; border-radius: 14px; backdrop-filter: blur(8px);
  box-shadow: 0 6px 22px rgba(0,0,0,.25);
}
div[data-testid="stMetricLabel"] p { color:#94a3b8; font-size:.8rem; text-transform:uppercase; letter-spacing:.6px; }
div[data-testid="stMetricValue"] { color:#fff; font-weight:600; font-size:1.55rem; }
div[data-testid="stMetricValue"] > div { white-space: normal !important; overflow: visible !important; text-overflow: clip !important; line-height:1.2; }

.loc-card {
  padding: 12px 16px; margin-bottom: 12px; border-radius: 14px;
  background: rgba(255,255,255,.045); border:1px solid rgba(255,255,255,.08);
  box-shadow: 0 6px 22px rgba(0,0,0,.25); backdrop-filter: blur(8px);
  transition: transform .15s ease, border-color .15s ease;
}
.loc-card:hover { transform: translateY(-3px); border-color: rgba(245,194,107,.45); }
.loc-card .name { color:#cbd5e1; font-weight:500; font-size:.9rem; }
.loc-card .num { font-size:30px; font-weight:700; color:#fff; line-height:1.15; }

/* tabs / buttons / inputs */
button[data-baseweb="tab"] { font-weight:500; }
button[data-baseweb="tab"][aria-selected="true"] { color:#f5c26b; }
.stButton > button, .stDownloadButton > button {
  border-radius: 10px; border:1px solid rgba(245,194,107,.45);
  background: linear-gradient(135deg, rgba(245,194,107,.18), rgba(56,189,248,.12));
  color:#fff; font-weight:500; transition: all .15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color:#f5c26b; box-shadow:0 0 18px rgba(245,194,107,.35); transform: translateY(-1px);
}
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg,#38bdf8,#6366f1); border:none;
}
div[data-testid="stDataFrame"], div[data-testid="stPlotlyChart"] {
  border-radius: 14px; border:1px solid rgba(255,255,255,.06);
  background: rgba(255,255,255,.025); padding: 4px;
}
div[data-testid="stFileUploader"] section {
  background: rgba(255,255,255,.03); border:1px dashed rgba(245,194,107,.45); border-radius:14px;
}
hr { border-color: rgba(255,255,255,.08); }
h2, h3 { font-family:'Playfair Display', serif; font-weight:600; color:#f1f5f9; }
</style>
"""


def inject():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title, subtitle=""):
    st.markdown(f"<div class='hero'><h1>{title}</h1><p>{subtitle}</p><div class='gold'></div></div>",
                unsafe_allow_html=True)
