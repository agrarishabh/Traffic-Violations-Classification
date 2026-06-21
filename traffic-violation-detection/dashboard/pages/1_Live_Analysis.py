"""dashboard/pages/1_Live_Analysis.py"""
import sys, cv2, numpy as np, io
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
from PIL import Image

from dashboard.theme import (inject_css, page_header, metric_row, metric_card,
                               section_title, divider, violation_card,
                               no_violation, conf_bar, live_badge)
inject_css()

@st.cache_resource(show_spinner=False)
def load_pipeline():
    from core.violation_detector import ViolationDetector
    from core.ocr                import LicensePlateOCR
    from core.evidence_generator import EvidenceGenerator
    vd = ViolationDetector(); vd.load_models()
    oc = LicensePlateOCR();   oc.load_models()
    gn = EvidenceGenerator(save_to_db=True)
    return vd, oc, gn

def _downscale(img, max_side=1280):
    """Cap the longest side to limit memory use during detection + OCR.

    Large uploads (e.g. 12 MP phone photos) cause EasyOCR's text detector
    to allocate huge intermediate tensors on CPU, which can OOM-kill the
    container. Downscaling keeps peak memory bounded with negligible
    accuracy loss for traffic-scene detection.
    """
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img
    scale = max_side / float(longest)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def run(image_bytes, fname):
    import gc
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None: return None,None,None,None
    img = _downscale(img, max_side=1280)
    vd,oc,gn = load_pipeline()
    vr  = vd.detect_all(img)
    dr  = vr.detection_result
    bbs = [v.bbox for v in dr.vehicles] if dr else []
    pl  = oc.extract(img, vehicle_bboxes=bbs)
    pno = pl[0].plate_text_clean if pl else None
    for v in vr.violations:
        if not v.plate_number and pno: v.plate_number = pno
    pkg = gn.generate(image=img, image_path=fname,
                      detection_result=dr, violation_result=vr, plate_results=pl)
    gc.collect()
    return img, vr, pl, pkg

# ════════════════════════════════════════════════════════════
page_header("Live Analysis", "Upload a traffic image for instant violation detection")

# Status bar
st.markdown(f"""
<div style="display:flex;align-items:center;gap:12px;
            background:#080808;border:1px solid rgba(219,226,220,0.12);
            border-radius:10px;padding:10px 18px;margin-bottom:20px">
    {live_badge()}
    <span style="color:#616161;font-size:0.8rem">
        YOLOv8 Detection &nbsp;&#183;&nbsp; EasyOCR &nbsp;&#183;&nbsp;
        Evidence Generator &nbsp;&#183;&nbsp; SQLite Storage
    </span>
</div>""", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Drop traffic image here or click to browse  (JPG / PNG / BMP / WebP — max 10 MB)",
    type=["jpg","jpeg","png","bmp","webp"])

if uploaded:
    raw  = uploaded.read()
    fname= uploaded.name
    c_orig, c_out = st.columns(2, gap="medium")
    with c_orig:
        section_title("ORIGINAL IMAGE")
        pil = Image.open(io.BytesIO(raw))
        st.image(pil, use_column_width=True,
                 caption=f"{fname}  {pil.width}x{pil.height}px")

    with st.spinner("Running AI analysis..."):
        img, vr, plates, pkg = run(raw, fname)

    if pkg is None:
        st.error("Could not decode image."); st.stop()

    with c_out:
        section_title("AI ANNOTATED OUTPUT")
        if pkg.annotated_image is not None:
            rgb = cv2.cvtColor(pkg.annotated_image, cv2.COLOR_BGR2RGB)
            st.image(rgb, use_column_width=True,
                     caption=f"ID: {pkg.evidence_id}")

    divider()
    metric_row([
        metric_card(len(vr.violations),      "Violations"),
        metric_card(f"Rs.{vr.total_fines:,}","Total Fine"),
        metric_card(len(plates),             "Plates Detected"),
        metric_card(f"{vr.processing_ms:.0f}ms", "Process Time"),
    ])
    divider()

    lft, rgt = st.columns(2, gap="large")
    with lft:
        section_title("VIOLATIONS DETECTED")
        if vr.violations:
            for v in vr.violations:
                st.markdown(violation_card(v.display_name, v.confidence,
                            v.fine_amount, v.plate_number, v.description),
                            unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background:rgba(219,226,220,0.06);border:1px solid rgba(219,226,220,0.15);
                        border-radius:10px;padding:14px 18px;margin-top:10px;text-align:center">
                <span style="color:#DBE2DC;font-weight:800;font-size:1.1rem">
                    TOTAL FINE: Rs.{vr.total_fines:,}
                </span>
                <span style="color:#424242;font-size:0.78rem;margin-left:12px">
                    ({len(vr.violations)} violation(s))
                </span>
            </div>""", unsafe_allow_html=True)
        else:
            no_violation()

    with rgt:
        section_title("LICENSE PLATES")
        if plates:
            for p in plates:
                vc = "#DBE2DC"
                st.markdown(f"""
                <div style="background:#080808;border:1px solid {vc}30;
                            border-radius:10px;padding:16px 20px;margin:8px 0">
                    <div style="display:flex;align-items:center;
                                justify-content:space-between;margin-bottom:10px">
                        <span class="plate">{p.plate_text_clean}</span>
                        <span style="font-size:0.72rem;font-weight:700;
                                     color:{vc};background:{vc}15;
                                     border:1px solid {vc}30;border-radius:12px;
                                     padding:3px 10px">
                            {'VALID' if p.is_valid_format else 'PARTIAL'}
                        </span>
                    </div>
                    {conf_bar('OCR Confidence', p.confidence)}
                    <div style="font-size:0.7rem;color:#424242;margin-top:6px">
                        method: {p.detection_method}
                    </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center;padding:32px;color:#424242;
                        background:#080808;border:1px solid #1a1a1a;border-radius:10px">
                <div style="font-size:0.85rem">No plates detected</div>
                <div style="font-size:0.75rem;margin-top:4px;color:#333">
                    Plate may be obscured or outside frame
                </div>
            </div>""", unsafe_allow_html=True)

        dr = vr.detection_result
        if dr:
            st.markdown("<br>", unsafe_allow_html=True)
            section_title("OBJECT DETECTION")
            st.markdown(
                conf_bar(f"Vehicles ({dr.vehicle_count})",
                         min(1.0, dr.vehicle_count/10))
                + conf_bar(f"Persons ({dr.person_count})",
                           min(1.0, dr.person_count/10))
                + conf_bar(f"Traffic Lights ({len(dr.traffic_lights)})",
                           min(1.0, len(dr.traffic_lights)/5)),
                unsafe_allow_html=True)

    divider()
    e1, e2 = st.columns([3,1])
    with e1:
        with st.expander("Evidence Package Details"):
            st.code(f"Evidence ID : {pkg.evidence_id}\n"
                    f"Timestamp  : {pkg.timestamp}\n"
                    f"Annotated  : {pkg.annotated_path}\n"
                    f"Metadata   : {pkg.metadata_path}\n"
                    f"DB rows    : {pkg.db_record_ids}")
    with e2:
        if pkg.annotated_image is not None:
            _, enc = cv2.imencode(".jpg", pkg.annotated_image)
            st.download_button("Download Annotated Image",
                               data=enc.tobytes(),
                               file_name=f"evidence_{pkg.evidence_id}.jpg",
                               mime="image/jpeg",
                               use_container_width=True)
else:
    st.markdown("""
    <div style="text-align:center;padding:72px 20px;
                border:1px dashed rgba(219,226,220,0.2);
                border-radius:14px;background:#080808">
        <div style="font-size:3rem;color:#DBE2DC;opacity:0.4">&#9685;</div>
        <h3 style="color:#e0e0e0;margin:16px 0 8px;font-weight:700">
            Upload a Traffic Image
        </h3>
        <p style="color:#424242;font-size:0.88rem">
            JPG &nbsp;&#183;&nbsp; PNG &nbsp;&#183;&nbsp; BMP &nbsp;&#183;&nbsp;
            WebP &nbsp;&#183;&nbsp; Max 10 MB
        </p>
        <div style="display:flex;gap:10px;justify-content:center;
                    flex-wrap:wrap;margin-top:20px">""" +
    "".join([f'<span style="background:#111;border:1px solid #1a1a1a;'
              f'border-radius:6px;padding:5px 12px;font-size:0.75rem;'
              f'color:#616161">{t}</span>'
              for t in ["Helmet","Seatbelt","Triple Riding",
                         "Red Light","Stop Line","Number Plate"]]) +
    "</div></div>", unsafe_allow_html=True)
