"""Downloadable PDF report."""
import io
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fpdf import FPDF

from config import LEVEL_COLORS


def _t(x):
    """fpdf core fonts are latin-1 only."""
    return str(x).encode("latin-1", "replace").decode("latin-1")


def _fig_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def status_chart(status):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    s = status.sort_values("Occupancy")
    ax.barh(s["Location"], s["Occupancy"] * 100, color=[LEVEL_COLORS[l] for l in s["Level"]])
    ax.set_xlabel("Occupancy (% of capacity)")
    ax.set_title("Current congestion by location")
    return _fig_png(fig)


def forecast_chart(details, location, horizon):
    d = details[(details.Location == location) & (details.Horizon == horizon)].sort_values("time")
    d = d.tail(288)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(d["time"], d["actual"], label="Actual", color="#1f77b4")
    ax.plot(d["time"], d["pred"], label=f"Predicted ({horizon} min ahead)", color="#d62839")
    ax.set_title(f"{location}: predicted vs actual (last 24 h of test period)")
    ax.legend()
    fig.autofmt_xdate()
    return _fig_png(fig)


def _table(pdf, df, widths, font=8):
    pdf.set_font("Helvetica", "B", font)
    for c, w in zip(df.columns, widths):
        pdf.cell(w, 6, _t(c)[:22], border=1, fill=True)
    pdf.ln()
    pdf.set_font("Helvetica", "", font)
    for _, row in df.iterrows():
        for v, w in zip(row, widths):
            pdf.cell(w, 6, _t(round(v, 2) if isinstance(v, float) else v)[:26], border=1)
        pdf.ln()


def build_pdf(status, alerts, metrics=None, details=None, thresholds=(0.25, 0.5, 0.75)):
    pdf = FPDF()
    pdf.set_auto_page_break(True, 15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "Campus Crowd Density & Congestion Report", ln=1)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Generated: {datetime.now():%Y-%m-%d %H:%M}", ln=1)
    pdf.cell(0, 6, f"Level thresholds (occupancy): Medium >= {thresholds[0]:.0%}, "
                   f"High >= {thresholds[1]:.0%}, Critical >= {thresholds[2]:.0%}", ln=1)
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "1. Current campus status", ln=1)
    pdf.set_fill_color(230, 230, 230)
    st = status[["Location", "Count", "Capacity", "Occupancy %", "Level"]]
    _table(pdf, st, [60, 25, 25, 35, 30], 9)
    pdf.ln(3)
    pdf.image(status_chart(status), w=170)

    pdf.add_page()
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "2. Recent alerts", ln=1)
    if alerts is None or alerts.empty:
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, "No alerts recorded.", ln=1)
    else:
        a = alerts.head(25).copy()
        a["ts"] = a["ts"].dt.strftime("%Y-%m-%d %H:%M")
        a = a[["ts", "location", "level", "count", "kind"]]
        a.columns = ["Time", "Location", "Level", "People", "Type"]
        _table(pdf, a, [38, 55, 25, 22, 30], 8)

    if metrics is not None and not metrics.empty:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "3. Prediction accuracy", ln=1)
        m = metrics[metrics.Location == "ALL CAMPUS"][
            ["Horizon", "Samples", "MAE (people)", "RMSE (people)", "WAPE %", "Improvement vs baseline %",
             "Level accuracy %"]].copy()
        m.columns = ["Horizon", "N", "MAE", "RMSE", "WAPE %", "Gain vs base %", "Level acc %"]
        _table(pdf, m, [22, 22, 24, 24, 24, 32, 28], 8)
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Per location (all horizons)", ln=1)
        p = metrics[metrics.Location != "ALL CAMPUS"][
            ["Location", "Horizon", "MAE (people)", "WAPE %", "Level accuracy %"]].copy()
        p.columns = ["Location", "Horizon", "MAE", "WAPE %", "Level acc %"]
        _table(pdf, p, [60, 22, 25, 25, 30], 8)
        if details is not None and not details.empty:
            top = metrics[metrics.Location != "ALL CAMPUS"].sort_values("MAE (people)").iloc[-1]
            pdf.ln(4)
            pdf.image(forecast_chart(details, top["Location"], int(top["Horizon"])), w=170)
    return bytes(pdf.output())
