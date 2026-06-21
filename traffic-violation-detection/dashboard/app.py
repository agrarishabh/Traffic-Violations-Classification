"""dashboard/app.py  —  Entrypoint (st.navigation router)

Uses the modern st.navigation / st.Page API instead of the legacy
`pages/` auto-discovery. This keeps the served document anchored at the
app root, so Streamlit's `_stcore/*` endpoints and websocket always
resolve correctly even when a subpage URL is loaded directly behind a
reverse proxy (Railway/Render/etc.).
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

st.set_page_config(page_title="TrafficAI", page_icon="◈",
                   layout="wide", initial_sidebar_state="expanded")

# Lightweight DB init only — heavy model loading happens lazily on first analysis
if "startup_done" not in st.session_state:
    try:
        from core.database import init_db
        init_db()
    except Exception:
        pass
    st.session_state["startup_done"] = True

# ── Page registry (paths relative to this entrypoint) ─────────
pages = [
    st.Page("home.py",                      title="Home",              icon="🏠", default=True),
    st.Page("pages/1_Live_Analysis.py",     title="Live Analysis",     icon="📹"),
    st.Page("pages/2_Violation_Records.py", title="Violation Records", icon="📋"),
    st.Page("pages/3_Analytics.py",         title="Analytics",         icon="📊"),
    st.Page("pages/4_Reports.py",           title="Reports",           icon="📄"),
]

pg = st.navigation(pages)
pg.run()
