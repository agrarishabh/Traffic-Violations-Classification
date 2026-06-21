"""dashboard/home.py  —  Home page (rendered via st.navigation)"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
from datetime import datetime

from dashboard.theme import (inject_css, page_header, metric_row,
                               metric_card, section_title, divider, live_badge)
# Compatibility shim — ignore extra kwargs if callers pass them
_mc_orig = metric_card
metric_card = lambda v, l, **kw: _mc_orig(v, l)
inject_css()

@st.cache_data(ttl=30)
def get_stats():
    try:
        from core.database import SessionLocal, init_db, get_stats as _g
        init_db(); db = SessionLocal(); s = _g(db); db.close(); return s
    except Exception:
        return {"total_images":0,"total_violations":0,"total_fines":0,"by_type":{}}

@st.cache_data(ttl=30)
def get_recent(n=6):
    try:
        from core.database import SessionLocal, init_db, get_violations
        init_db(); db = SessionLocal()
        r = get_violations(db,skip=0,limit=n); db.close()
        return [x.to_dict() for x in r]
    except Exception: return []

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:24px 0 16px;border-bottom:1px solid rgba(219,226,220,0.12)">
        <div style="font-size:2rem;color:#DBE2DC;font-family:'JetBrains Mono',monospace;
                    font-weight:700;letter-spacing:0.05em">◈ TrafficAI</div>
        <div style="font-size:0.7rem;color:#424242;letter-spacing:0.15em;
                    text-transform:uppercase;margin-top:4px">
            Violation Detection System
        </div>
    </div>""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;padding:48px 0 24px">
    <div style="font-size:0.72rem;font-weight:700;letter-spacing:0.25em;
                color:#DBE2DC;text-transform:uppercase;margin-bottom:16px">
        AI-POWERED TRAFFIC ENFORCEMENT
    </div>
    <div class="hero-title">Traffic Violation Detection</div>
    <p style="color:#616161;font-size:0.92rem;margin:14px 0 20px">
        Automated detection &nbsp;&#183;&nbsp; Classification &nbsp;&#183;&nbsp;
        Evidence Generation &nbsp;&#183;&nbsp; Analytics
    </p>
    {live_badge()}
</div>""", unsafe_allow_html=True)

# ── Metrics ───────────────────────────────────────────────────
s   = get_stats()
avg = round(s["total_violations"] / max(1, s["total_images"]), 1)
metric_row([
    metric_card(s["total_images"],          "Images Processed"),
    metric_card(s["total_violations"],      "Violations Found"),
    metric_card(f"Rs.{s['total_fines']:,}", "Total Fines"),
    metric_card(avg,                        "Avg per Image"),
])
divider()

# ── Two columns ───────────────────────────────────────────────
col1, col2 = st.columns(2, gap="large")

with col1:
    section_title("DETECTABLE VIOLATIONS")
    items = [
        ("Helmet Non-Compliance",   "Rider without helmet on two-wheeler",   "cyan"),
        ("Seatbelt Non-Compliance", "Driver without seatbelt in vehicle",    "cyan"),
        ("Triple Riding",           "More than 2 persons on two-wheeler",    "cyan"),
        ("Red Light Violation",     "Vehicle crossing red signal",           "cyan"),
        ("Stop Line Violation",     "Vehicle crosses stop line",             "cyan"),
        ("Wrong-Side Driving",      "Vehicle on incorrect lane side",        "cyan"),
        ("Illegal Parking",         "Parked in restricted zone",             "cyan"),
    ]
    for name, desc, col in items:
        c = "#DBE2DC"
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;padding:10px 14px;
                    margin:4px 0;background:#111;border:1px solid #1a1a1a;
                    border-left:2px solid {c};border-radius:8px;transition:all 0.2s">
            <div style="flex:1">
                <div style="font-size:0.86rem;font-weight:600;color:#e0e0e0">{name}</div>
                <div style="font-size:0.74rem;color:#616161;margin-top:2px">{desc}</div>
            </div>
            <span style="font-size:0.68rem;font-weight:700;color:{c};
                         background:{'rgba(255,23,68,0.1)' if col=='red' else 'rgba(219,226,220,0.1)'};
                         border:1px solid {'rgba(255,23,68,0.3)' if col=='red' else 'rgba(219,226,220,0.3)'};
                         border-radius:12px;padding:2px 9px;letter-spacing:0.08em">
                ACTIVE
            </span>
        </div>""", unsafe_allow_html=True)

with col2:
    section_title("RECENT VIOLATIONS")
    recent = get_recent(6)
    if recent:
        for r in recent:
            dt = (r.get("detected_at","")[:16]).replace("T"," ")
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:12px;
                        padding:10px 14px;margin:4px 0;
                        background:rgba(219,226,220,0.04);
                        border:1px solid rgba(219,226,220,0.12);
                        border-left:2px solid #DBE2DC;border-radius:8px">
                <div style="flex:1">
                    <div style="font-size:0.85rem;font-weight:600;color:#edf0ed">
                        {r.get('display_name','Unknown')}</div>
                    <div style="font-size:0.72rem;color:#616161;margin-top:2px">{dt}</div>
                </div>
                <div style="text-align:right">
                    <div style="font-size:0.82rem;font-weight:700;color:#DBE2DC">
                        Rs.{r.get('fine_amount',0):,}</div>
                    <div style="font-size:0.68rem;color:#424242">
                        {r.get('confidence',0):.0%}</div>
                </div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:40px;color:#424242;
                    background:#111;border:1px solid #1a1a1a;border-radius:10px">
            <div style="font-size:0.85rem">No violations recorded yet</div>
            <div style="font-size:0.75rem;margin-top:6px;color:#333">
                Use Live Analysis to process images
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Refresh Stats", use_container_width=True):
        st.cache_data.clear(); st.rerun()

divider()

# ── Nav grid ─────────────────────────────────────────────────
section_title("NAVIGATION")
n1,n2,n3,n4 = st.columns(4, gap="medium")
for col, title, desc, border in [
    (n1,"Live Analysis","Upload & detect violations in real time","#DBE2DC"),
    (n2,"Violation Records","Search all stored violation records","#c4cdc5"),
    (n3,"Analytics","Interactive charts and trend analysis","#DBE2DC"),
    (n4,"Reports","Export CSV or printable text report","#c4cdc5"),
]:
    with col:
        st.markdown(f"""
        <div style="background:#111;border:1px solid {border}30;
                    border-top:2px solid {border};border-radius:10px;
                    padding:18px;text-align:center;
                    transition:all 0.3s;cursor:pointer">
            <div style="font-weight:700;color:#e0e0e0;font-size:0.88rem;
                        margin-bottom:8px">{title}</div>
            <div style="color:#616161;font-size:0.75rem;line-height:1.5">{desc}</div>
        </div>""", unsafe_allow_html=True)

st.markdown(f"""
<div style="text-align:center;color:#2a2a2a;font-size:0.72rem;
            padding:24px 0 8px;letter-spacing:0.06em">
    TrafficAI v1.0 &nbsp;&#183;&nbsp; Flipkart Gridlock Hackathon &nbsp;&#183;&nbsp;
    {datetime.now().strftime('%Y-%m-%d %H:%M')}
</div>""", unsafe_allow_html=True)
