"""dashboard/pages/2_Violation_Records.py"""
import sys, cv2
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st, pandas as pd
from datetime import datetime, timedelta

from dashboard.theme import (inject_css, page_header, metric_row,
                               metric_card, section_title, divider, conf_bar)
inject_css()

@st.cache_data(ttl=15)
def fetch(vtype, plate, dfrom, dto):
    from core.database import SessionLocal, init_db, get_violations
    init_db(); db = SessionLocal()
    r = get_violations(db,skip=0,limit=500,violation_type=vtype or None,
                       plate_number=plate or None,date_from=dfrom,date_to=dto)
    db.close(); return [x.to_dict() for x in r]

page_header("Violation Records","Search, filter and review all recorded violations")

with st.sidebar:
    st.markdown("""
    <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.15em;
                color:#DBE2DC;text-transform:uppercase;padding:12px 0 8px;
                border-bottom:1px solid rgba(219,226,220,0.1);margin-bottom:14px">
        Filters
    </div>""", unsafe_allow_html=True)
    from config import VIOLATION_DISPLAY_NAMES
    tmap = {"All Types":""} | {v:k for k,v in VIOLATION_DISPLAY_NAMES.items()}
    sel  = st.selectbox("Violation Type", list(tmap.keys()))
    plate_q = st.text_input("Plate Number", placeholder="e.g. MH12")
    c1,c2 = st.columns(2)
    df_inp = c1.date_input("From", value=datetime.now()-timedelta(days=30))
    dt_inp = c2.date_input("To",   value=datetime.now())
    st.markdown("<br>",unsafe_allow_html=True)
    if st.button("Apply Filters", use_container_width=True, type="primary"):
        st.cache_data.clear()
    if st.button("Reset", use_container_width=True):
        st.cache_data.clear(); st.rerun()

records = fetch(tmap[sel], plate_q,
                datetime.combine(df_inp, datetime.min.time()),
                datetime.combine(dt_inp, datetime.max.time()))

total_fine    = sum(r.get("fine_amount",0) for r in records)
unique_plates = len({r["plate_number"] for r in records if r.get("plate_number")})
avg_conf      = sum(r.get("confidence",0) for r in records)/max(1,len(records))

metric_row([
    metric_card(len(records),        "Records Found"),
    metric_card(f"Rs.{total_fine:,}","Total Fines"),
    metric_card(unique_plates,       "Unique Plates"),
    metric_card(f"{avg_conf:.0%}",   "Avg Confidence"),
])
divider()

if not records:
    st.markdown("""
    <div style="text-align:center;padding:60px;background:#080808;
                border:1px solid #1a1a1a;border-radius:12px;color:#424242">
        <div style="font-size:1.5rem;color:#333">&#9651;</div>
        <div style="margin-top:10px;font-size:0.88rem">No records match the filters</div>
    </div>""", unsafe_allow_html=True); st.stop()

rows = [{"ID":r["id"],"Violation":r["display_name"],
         "Conf":f"{r['confidence']:.0%}","Plate":r.get("plate_number") or "—",
         "Fine":r["fine_amount"],"Image":r.get("image_filename") or "—",
         "Detected At":(r.get("detected_at","")[:16]).replace("T"," ")}
        for r in records]
df = pd.DataFrame(rows)

section_title(f"RECORDS  ({len(records)} FOUND)")
st.dataframe(df, use_container_width=True, hide_index=True,
             column_config={
                 "ID":   st.column_config.NumberColumn(width="small"),
                 "Fine": st.column_config.NumberColumn(format="Rs.%d",width="small"),
                 "Conf": st.column_config.TextColumn(width="small"),
             })
divider()
section_title("RECORD DETAIL")
sel_id = st.number_input("Enter Record ID", min_value=1, step=1,
                          value=records[0]["id"])
match  = next((r for r in records if r["id"]==sel_id), None)

if match:
    d1,d2,d3 = st.columns(3, gap="medium")
    conf = match.get("confidence",0)
    fine = match.get("fine_amount",0)

    with d1:
        st.markdown(f"""
        <div style="background:#080808;border:1px solid rgba(219,226,220,0.15);
                    border-top:2px solid #DBE2DC;border-radius:10px;padding:20px">
            <div style="font-size:0.68rem;font-weight:700;color:#DBE2DC;
                        letter-spacing:0.12em;text-transform:uppercase;
                        margin-bottom:12px">Violation</div>
            <div style="font-size:1rem;font-weight:700;color:#edf0ed;
                        margin-bottom:12px">&#9888; {match['display_name']}</div>
            {conf_bar('Confidence', conf)}
            <div style="margin-top:12px;font-size:0.8rem;color:#616161;line-height:1.8">
                <div><b style="color:#424242">Fine:</b>
                     <span style="color:#DBE2DC;font-weight:700"> Rs.{fine:,}</span></div>
                <div><b style="color:#424242">Record ID:</b> {match['id']}</div>
                <div><b style="color:#424242">Detected:</b>
                     {(match.get('detected_at','')[:16]).replace('T',' ')}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    with d2:
        plate = match.get("plate_number") or ""
        st.markdown(f"""
        <div style="background:#080808;border:1px solid rgba(219,226,220,0.15);
                    border-top:2px solid #DBE2DC;border-radius:10px;padding:20px">
            <div style="font-size:0.68rem;font-weight:700;color:#DBE2DC;
                        letter-spacing:0.12em;text-transform:uppercase;
                        margin-bottom:12px">Vehicle</div>
            {'<span class="plate">' + plate + '</span>' if plate else
             '<div style="color:#424242;font-size:0.82rem">Plate not detected</div>'}
            <div style="margin-top:12px;font-size:0.8rem;color:#616161;line-height:1.8">
                <div><b style="color:#424242">Image:</b>
                     {match.get('image_filename') or '—'}</div>
            </div>
            <div style="margin-top:10px;background:#111;border-radius:8px;
                        padding:10px;font-size:0.75rem;color:#9e9e9e;line-height:1.5">
                {match.get('description') or '—'}
            </div>
        </div>""", unsafe_allow_html=True)

    with d3:
        ann = match.get("annotated_path","")
        st.markdown("""
        <div style="background:#080808;border:1px solid #1a1a1a;
                    border-top:2px solid #333;border-radius:10px;padding:20px">
            <div style="font-size:0.68rem;font-weight:700;color:#616161;
                        letter-spacing:0.12em;text-transform:uppercase;
                        margin-bottom:12px">Evidence Image</div>""",
        unsafe_allow_html=True)
        if ann and Path(ann).exists():
            img = cv2.imread(ann)
            if img is not None:
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                         use_column_width=True)
        else:
            st.markdown("""
            <div style="text-align:center;padding:24px;color:#333">
                <div style="font-size:1.5rem">&#9685;</div>
                <div style="font-size:0.78rem;margin-top:6px">No image found</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
