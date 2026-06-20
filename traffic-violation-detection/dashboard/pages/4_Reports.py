"""dashboard/pages/4_Reports.py"""
import sys, io
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(page_title="Reports  TrafficAI", page_icon="◈", layout="wide")
from dashboard.theme import (inject_css, page_header, metric_row,
                               metric_card, section_title, divider)
inject_css()

@st.cache_data(ttl=30)
def fetch_all(dfrom, dto):
    from core.database import SessionLocal, init_db, get_violations, get_stats
    init_db(); db = SessionLocal()
    r = get_violations(db, skip=0, limit=50000, date_from=dfrom, date_to=dto)
    s = get_stats(db); db.close()
    return [x.to_dict() for x in r], s

def to_df(records):
    if not records: return pd.DataFrame()
    return pd.DataFrame([{
        "ID":          r["id"],
        "Violation":   r["display_name"],
        "Confidence":  round(r["confidence"], 4),
        "Plate":       r.get("plate_number") or "",
        "Fine (INR)":  r["fine_amount"],
        "Image":       r.get("image_filename") or "",
        "Detected At": r.get("detected_at") or "",
    } for r in records])

# ══════════════════════════════════════════════════════════════
page_header("Reports", "Generate and export violation reports for any date range")

# ── Period selector ───────────────────────────────────────────
st.markdown("""
<div style="background:#080808;border:1px solid rgba(219,226,220,0.12);
            border-radius:12px;padding:20px 24px;margin-bottom:20px">
    <div style="font-size:0.68rem;font-weight:700;color:#DBE2DC;
                letter-spacing:0.15em;text-transform:uppercase;
                margin-bottom:14px">Report Period</div>""",
unsafe_allow_html=True)

c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    date_from = st.date_input("From", value=datetime.now()-timedelta(days=30),
                               key="rpt_from")
with c2:
    date_to = st.date_input("To", value=datetime.now(), key="rpt_to")
with c3:
    preset = st.selectbox("Quick Select",
                           ["Custom","Today","Last 7 Days","Last 30 Days","All Time"],
                           key="rpt_preset")
    if preset == "Today":
        date_from = date_to = datetime.now().date()
    elif preset == "Last 7 Days":
        date_from = (datetime.now()-timedelta(days=7)).date()
        date_to   = datetime.now().date()
    elif preset == "Last 30 Days":
        date_from = (datetime.now()-timedelta(days=30)).date()
        date_to   = datetime.now().date()
    elif preset == "All Time":
        date_from = datetime(2020, 1, 1).date()
        date_to   = datetime.now().date()

st.markdown("</div>", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────
dt_from = datetime.combine(date_from, datetime.min.time())
dt_to   = datetime.combine(date_to,   datetime.max.time())
records, stats = fetch_all(dt_from, dt_to)
df = to_df(records)
total_fine    = sum(r["fine_amount"] for r in records)
unique_plates = len({r["plate_number"] for r in records if r.get("plate_number")})

# ── Metrics ───────────────────────────────────────────────────
metric_row([
    metric_card(len(records),                    "Total Violations"),
    metric_card(f"Rs.{total_fine:,}",            "Total Fines"),
    metric_card(unique_plates,                   "Unique Plates"),
    metric_card(stats.get("total_images", 0),    "Images Processed"),
])
divider()

if df.empty:
    st.markdown("""
    <div style="text-align:center;padding:60px;background:#080808;
                border:1px solid #1a1a1a;border-radius:12px;color:#424242">
        No data for the selected period.
    </div>""", unsafe_allow_html=True)
    st.stop()

# ── Breakdown table ───────────────────────────────────────────
section_title("BREAKDOWN BY VIOLATION TYPE")
bk = (df.groupby("Violation")
      .agg(Count=("ID","count"), Fine=("Fine (INR)","sum"), Conf=("Confidence","mean"))
      .reset_index().sort_values("Count", ascending=False))
bk["Avg Conf"] = bk["Conf"].apply(lambda x: f"{x:.0%}")
bk = bk.rename(columns={"Fine":"Total Fine (Rs.)"}).drop(columns=["Conf"])
st.dataframe(bk, use_container_width=True, hide_index=True,
             column_config={
                 "Total Fine (Rs.)": st.column_config.NumberColumn(format="Rs.%d")})
divider()

# ── Downloads ─────────────────────────────────────────────────
section_title("DOWNLOAD")
dl1, dl2 = st.columns(2, gap="large")

with dl1:
    st.markdown("""
    <div style="background:#080808;border:1px solid rgba(219,226,220,0.15);
                border-radius:10px;padding:18px;text-align:center;margin-bottom:10px">
        <div style="font-weight:700;color:#e0e0e0;margin-bottom:4px">Full CSV Export</div>
        <div style="font-size:0.75rem;color:#424242">All columns</div>
    </div>""", unsafe_allow_html=True)
    buf = io.StringIO()
    df.to_csv(buf, index=False, encoding="utf-8")
    st.download_button("Download CSV",
                       data=buf.getvalue().encode("utf-8"),
                       file_name=f"violations_{date_from}_{date_to}.csv",
                       mime="text/csv", use_container_width=True)
    st.caption(f"{len(df)} records")

with dl2:
    st.markdown("""
    <div style="background:#080808;border:1px solid rgba(219,226,220,0.12);
                border-radius:10px;padding:18px;text-align:center;margin-bottom:10px">
        <div style="font-weight:700;color:#e0e0e0;margin-bottom:4px">Text Summary</div>
        <div style="font-size:0.75rem;color:#424242">Printable report</div>
    </div>""", unsafe_allow_html=True)
    lines = [
        "=" * 55,
        "  TRAFFIC VIOLATION DETECTION SYSTEM",
        "  Summary Report",
        "=" * 55,
        f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"  Period    : {date_from} to {date_to}",
        f"  Total     : {len(records)} violations",
        f"  Fines     : Rs.{total_fine:,}",
        f"  Images    : {stats.get('total_images', 0)}",
        "", "  BREAKDOWN", "-" * 55,
    ]
    for _, row in bk.iterrows():
        lines.append(f"  {row['Violation']:<32} {row['Count']:>4}x")
    st.download_button("Download Report",
                       data="\n".join(lines).encode("utf-8"),
                       file_name=f"report_{date_from}_{date_to}.txt",
                       mime="text/plain", use_container_width=True)

divider()

# ── Data preview ──────────────────────────────────────────────
section_title(f"DATA PREVIEW  (FIRST 50 OF {len(df)})")
st.dataframe(
    df[["ID","Violation","Confidence","Plate","Fine (INR)","Detected At"]].head(50),
    use_container_width=True, hide_index=True,
    column_config={
        "Fine (INR)": st.column_config.NumberColumn(format="Rs.%d"),
    },
)
if len(df) > 50:
    st.caption(f"Showing 50 of {len(df)}. Download CSV for full data.")
