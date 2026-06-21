"""dashboard/pages/3_Analytics.py"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st, pandas as pd
import plotly.express as px, plotly.graph_objects as go
from datetime import datetime

from dashboard.theme import (inject_css, page_header, metric_row,
                               metric_card, section_title, divider)
inject_css()

# ── Plotly theme helpers ──────────────────────────────────────
# DARK_BASE: only non-axis keys — avoids duplicate kwarg error
DARK_BASE = dict(
    plot_bgcolor  = "rgba(0,0,0,0)",
    paper_bgcolor = "rgba(0,0,0,0)",
    font          = dict(color="#9e9e9e", family="Inter"),
    margin        = dict(t=30, b=20, l=10, r=10),
)
# Reusable axis and legend dicts — never put in DARK_BASE
AXIS    = dict(gridcolor="#1a1a1a", linecolor="#1a1a1a",
               zerolinecolor="#1a1a1a", tickcolor="#424242")
LEG_H   = dict(orientation="h", y=-0.35, bgcolor="rgba(0,0,0,0)", bordercolor="#1a1a1a")
LEG_V   = dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1a1a1a")

CYAN   = "#DBE2DC"
CYAN2  = "#c4cdc5"
CYAN3  = "#9aab9c"
# Cyan-only shades for multi-series charts
SHADES = ["#DBE2DC","#c4cdc5","#9aab9c","#708573","#edf0ed","#4a5e4d","#f5f7f5","#b8c4b9"]

@st.cache_data(ttl=30)
def load():
    from core.database import SessionLocal, init_db, get_violations
    init_db(); db = SessionLocal()
    r = get_violations(db, skip=0, limit=10000); db.close()
    return [x.to_dict() for x in r]

def build(records):
    if not records: return pd.DataFrame()
    rows = []
    for r in records:
        s = r.get("detected_at", "") or ""
        try:    dt = datetime.fromisoformat(s)
        except: dt = datetime.now()
        rows.append({"vtype": r["violation_type"], "dname": r["display_name"],
                     "conf": r["confidence"], "fine": r["fine_amount"],
                     "plate": r.get("plate_number") or "Unknown",
                     "dt": dt, "date": dt.date(), "hour": dt.hour})
    return pd.DataFrame(rows)

page_header("Analytics", "Real-time violation trends and statistics")

c_ref, _ = st.columns([1, 5])
with c_ref:
    if st.button("Refresh", use_container_width=True):
        st.cache_data.clear(); st.rerun()

records = load()
df      = build(records)

if df.empty:
    st.markdown("""<div style="text-align:center;padding:80px;background:#0d0d0d;
    border:1px solid #1a1a1a;border-radius:12px;color:#424242">
    No data yet. Process images on Live Analysis first.</div>""",
    unsafe_allow_html=True); st.stop()

type_agg = (df.groupby("dname")
            .agg(count=("conf","count"), fine=("fine","sum"), conf=("conf","mean"))
            .reset_index().sort_values("count", ascending=False))

metric_row([
    metric_card(df["date"].nunique(),           "Active Days"),
    metric_card(len(df),                         "Total Violations"),
    metric_card(f"Rs.{df['fine'].sum():,.0f}",  "Total Fines"),
    metric_card(f"{df['conf'].mean():.0%}",      "Avg Confidence"),
])
divider()

# ── Row 1: Donut + Bar ────────────────────────────────────────
r1, r2 = st.columns(2, gap="medium")
with r1:
    section_title("DISTRIBUTION BY TYPE")
    fig1 = px.pie(type_agg, values="count", names="dname",
                  hole=0.58, color_discrete_sequence=SHADES)
    fig1.update_traces(textposition="outside", textfont_size=10,
                       marker=dict(line=dict(color="#000000", width=2)))
    fig1.update_layout(**DARK_BASE, showlegend=False, legend=LEG_V,
                       annotations=[dict(
                           text=f"<b>{len(df)}</b>",
                           x=0.5, y=0.5,
                           font=dict(size=20, color=CYAN),
                           showarrow=False)])
    st.plotly_chart(fig1, use_container_width=True)

with r2:
    section_title("COUNT AND FINE BY TYPE")
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        name="Count", x=type_agg["dname"], y=type_agg["count"],
        marker=dict(color=CYAN, opacity=0.80,
                    line=dict(color="#000000", width=1)),
        yaxis="y1",
    ))
    fig2.add_trace(go.Scatter(
        name="Fine (Rs.)", x=type_agg["dname"], y=type_agg["fine"],
        mode="lines+markers",
        line=dict(color=CYAN2, width=2, dash="dot"),
        marker=dict(size=7, color=CYAN2),
        yaxis="y2",
    ))
    fig2.update_layout(
        **DARK_BASE,
        xaxis  = dict(tickangle=-25, **AXIS),
        yaxis  = dict(title="Count",      **AXIS),
        yaxis2 = dict(title="Fine (Rs.)", overlaying="y", side="right", **AXIS),
        legend = LEG_V,
    )
    st.plotly_chart(fig2, use_container_width=True)

divider()
# ── Row 2: Trend + Hour ───────────────────────────────────────
r3, r4 = st.columns(2, gap="medium")
with r3:
    section_title("DAILY TREND")
    daily = df.groupby("date").agg(count=("vtype","count"),
                                    fine=("fine","sum")).reset_index()
    fig3  = go.Figure()
    fig3.add_trace(go.Scatter(
        x=daily["date"], y=daily["count"], name="Violations",
        fill="tozeroy", fillcolor="rgba(219,226,220,0.06)",
        line=dict(color=CYAN, width=2), mode="lines+markers",
        marker=dict(size=5, color=CYAN), yaxis="y1",
    ))
    fig3.add_trace(go.Scatter(
        x=daily["date"], y=daily["fine"], name="Fine (Rs.)",
        line=dict(color=CYAN2, width=2, dash="dot"),
        mode="lines+markers", marker=dict(size=5, color=CYAN2), yaxis="y2",
    ))
    fig3.update_layout(
        **DARK_BASE,
        xaxis  = dict(**AXIS),
        yaxis  = dict(title="Violations", **AXIS),
        yaxis2 = dict(title="Fine (Rs.)", overlaying="y", side="right", **AXIS),
        legend = LEG_V,
    )
    st.plotly_chart(fig3, use_container_width=True)

with r4:
    section_title("HOUR OF DAY PATTERN")
    ha   = df.groupby(["hour","dname"]).size().reset_index(name="count")
    fig4 = px.bar(ha, x="hour", y="count", color="dname",
                  color_discrete_sequence=SHADES, barmode="stack")
    fig4.update_layout(
        **DARK_BASE,
        xaxis  = dict(title="Hour", tickmode="linear", dtick=2, **AXIS),
        yaxis  = dict(title="Count", **AXIS),
        legend = LEG_H,
    )
    st.plotly_chart(fig4, use_container_width=True)

divider()
# ── Row 3: Confidence + Plates ───────────────────────────────
r5, r6 = st.columns([3, 2], gap="medium")
with r5:
    section_title("CONFIDENCE DISTRIBUTION")
    fig5 = px.histogram(df, x="conf", nbins=20, color="dname",
                        color_discrete_sequence=SHADES,
                        opacity=0.8, barmode="overlay")
    fig5.update_layout(
        **DARK_BASE,
        xaxis  = dict(title="Confidence", tickformat=".0%", **AXIS),
        yaxis  = dict(title="Count", **AXIS),
        legend = LEG_H,
    )
    st.plotly_chart(fig5, use_container_width=True)

with r6:
    section_title("TOP OFFENDING PLATES")
    pdf = (df[df["plate"] != "Unknown"]
           .groupby("plate")
           .agg(v=("vtype","count"), f=("fine","sum"))
           .reset_index().sort_values("v", ascending=False).head(8))
    if not pdf.empty:
        mx = pdf["v"].max()
        for i, (_, row) in enumerate(pdf.iterrows()):
            pct = int(row["v"] / mx * 100)
            rank = f"#{i+1}"
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;padding:8px 12px;
                        margin:3px 0;background:#0d0d0d;border:1px solid #1a1a1a;
                        border-radius:8px">
                <span style="color:#DBE2DC;font-size:0.8rem;min-width:22px;
                             font-weight:700">{rank}</span>
                <span class="plate" style="font-size:0.7rem;padding:3px 10px">
                    {row['plate']}</span>
                <div style="flex:1;margin:0 8px">
                    <div style="height:3px;background:#1a1a1a;border-radius:2px">
                        <div style="width:{pct}%;height:100%;
                                    background:#DBE2DC;border-radius:2px"></div>
                    </div>
                </div>
                <span style="font-size:0.75rem;color:#DBE2DC;font-weight:700">
                    {int(row['v'])}x</span>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#424242;padding:20px;text-align:center;'
                    'font-size:0.85rem">No plate data yet</div>',
                    unsafe_allow_html=True)

divider()
section_title("SUMMARY TABLE")
summary = type_agg.rename(columns={"dname": "Violation", "count": "Count",
                                    "fine": "Total Fine (Rs.)", "conf": "Avg Conf"})
summary["Avg Conf"] = summary["Avg Conf"].apply(lambda x: f"{x:.0%}")
st.dataframe(summary, use_container_width=True, hide_index=True)
