"""dashboard/theme.py  —  BLACK + CYAN only"""
MASTER_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
:root {
    --bg0:#000000; --bg1:#050505; --bg2:#0d0d0d; --bg3:#141414; --bg4:#1c1c1c;
    --cyan:#DBE2DC; --cyan2:#c4cdc5; --cyan3:#9aab9c;
    --ca15:rgba(219,226,220,0.15); --ca08:rgba(219,226,220,0.08); --ca04:rgba(219,226,220,0.04);
    --w1:#e0e0e0; --w2:#9e9e9e; --w3:#616161; --w4:#2a2a2a;
    --r:10px; --r2:14px;
}
.stApp{background:var(--bg0)!important;font-family:'Inter',sans-serif!important;color:var(--w1)!important;}
.stApp>header{background:transparent!important;}
section[data-testid="stSidebar"]{background:#020202!important;border-right:1px solid var(--ca08)!important;}
section[data-testid="stSidebar"] *{color:var(--w2)!important;}
#MainMenu,footer,header{visibility:hidden;}
::-webkit-scrollbar{width:4px;height:4px;}
::-webkit-scrollbar-track{background:var(--bg0);}
::-webkit-scrollbar-thumb{background:#9aab9c;border-radius:2px;}

@keyframes fadeUp{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:translateY(0)}}
@keyframes fadeLeft{from{opacity:0;transform:translateX(-18px)}to{opacity:1;transform:translateX(0)}}
@keyframes cyanFlow{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0.3}}

.hero-title{
    font-size:2.8rem;font-weight:900;letter-spacing:-0.02em;
    background:linear-gradient(90deg,#DBE2DC,#ffffff,#c4cdc5);
    background-size:200%;-webkit-background-clip:text;
    -webkit-text-fill-color:transparent;background-clip:text;
    animation:cyanFlow 5s ease infinite,fadeUp 0.6s ease-out;margin:0;
}
.metric-grid{display:flex;gap:12px;flex-wrap:wrap;margin:14px 0;}
.metric-card{
    flex:1;min-width:140px;background:var(--bg2);
    border:1px solid var(--bg4);border-top:2px solid var(--cyan);
    border-radius:var(--r2);padding:18px 20px;text-align:center;
    transition:all 0.3s;animation:fadeUp 0.5s ease-out;cursor:default;
}
.metric-card:hover{transform:translateY(-3px);border-color:var(--ca15);
    box-shadow:0 0 18px rgba(0,229,255,0.12);}
.metric-card .m-val{font-size:1.9rem;font-weight:800;color:var(--cyan);line-height:1;}
.metric-card .m-lbl{font-size:0.7rem;font-weight:600;color:var(--w2);
    text-transform:uppercase;letter-spacing:0.1em;margin-top:6px;}

.v-card{
    background:var(--ca04);border:1px solid var(--ca15);
    border-left:2px solid var(--cyan);border-radius:var(--r);
    padding:12px 16px;margin:5px 0;
    display:flex;align-items:center;gap:14px;
    animation:fadeLeft 0.4s ease-out;transition:all 0.25s;
}
.v-card:hover{background:var(--ca08);box-shadow:0 0 14px rgba(0,229,255,0.1);}
.v-card .v-name{font-size:0.9rem;font-weight:600;color:#80deea;}
.v-card .v-desc{font-size:0.74rem;color:var(--w3);margin-top:2px;}
.v-card .v-fine{margin-left:auto;white-space:nowrap;background:var(--ca08);
    border:1px solid var(--ca15);border-radius:20px;padding:4px 12px;
    font-size:0.75rem;font-weight:700;color:var(--cyan);}
.v-card .v-conf{font-size:0.68rem;color:var(--w3);text-align:right;margin-top:3px;}

.safe-card{background:var(--ca04);border:1px solid var(--ca15);
    border-radius:var(--r2);padding:28px;text-align:center;animation:fadeUp 0.5s ease-out;}
.safe-card .s-text{color:var(--cyan);font-weight:700;font-size:1rem;margin-top:10px;}
.safe-card .s-sub{color:var(--w3);font-size:0.8rem;margin-top:4px;}

.sec-title{font-size:0.68rem;font-weight:700;letter-spacing:0.15em;
    text-transform:uppercase;color:var(--cyan);
    display:flex;align-items:center;gap:8px;
    margin:20px 0 10px;padding-bottom:7px;
    border-bottom:1px solid var(--ca08);}
.sec-title::before{content:'';width:3px;height:12px;background:var(--cyan);
    border-radius:2px;display:inline-block;}

.plate{display:inline-block;background:var(--bg0);border:2px solid var(--cyan);
    border-radius:6px;padding:5px 14px;font-family:'JetBrains Mono',monospace;
    font-size:1rem;font-weight:700;color:var(--cyan);letter-spacing:0.12em;
    box-shadow:0 0 10px rgba(0,229,255,0.2);}

.cbar-wrap{margin:5px 0;}
.cbar-row{display:flex;justify-content:space-between;
    font-size:0.74rem;color:var(--w3);margin-bottom:3px;}
.cbar-track{height:4px;background:var(--bg4);border-radius:2px;overflow:hidden;}
.cbar-fill{height:100%;border-radius:2px;
    background:linear-gradient(90deg,var(--cyan2),var(--cyan));
    transition:width 0.9s ease-out;}

.div{height:1px;background:linear-gradient(90deg,transparent,var(--ca15),transparent);margin:18px 0;}
.live{display:inline-flex;align-items:center;gap:6px;background:var(--ca04);
    border:1px solid var(--ca15);border-radius:20px;padding:4px 12px;
    font-size:0.68rem;font-weight:700;color:var(--cyan);
    text-transform:uppercase;letter-spacing:0.12em;}
.live .dot{width:5px;height:5px;border-radius:50%;background:var(--cyan);animation:blink 1.5s infinite;}

.stButton>button{background:transparent!important;border:1px solid var(--cyan)!important;
    color:var(--cyan)!important;border-radius:var(--r)!important;font-weight:600!important;
    font-family:'Inter',sans-serif!important;transition:all 0.25s!important;}
.stButton>button:hover{background:var(--ca08)!important;
    box-shadow:0 0 14px rgba(0,229,255,0.25)!important;transform:translateY(-1px)!important;}
div[data-testid="stDataFrame"]{background:var(--bg2)!important;
    border:1px solid var(--bg4)!important;border-radius:var(--r)!important;}
.stSelectbox>div>div,.stTextInput>div>div{background:var(--bg2)!important;
    border-color:var(--bg4)!important;color:var(--w1)!important;border-radius:var(--r)!important;}
.stMetric{background:var(--bg2)!important;border:1px solid var(--bg4)!important;
    border-radius:var(--r)!important;padding:14px!important;}
.stMetric label{color:var(--w2)!important;font-size:0.74rem!important;}
.stMetric [data-testid="stMetricValue"]{color:var(--cyan)!important;font-weight:800!important;}
.stTabs [data-baseweb="tab-list"]{background:var(--bg2)!important;
    border-radius:var(--r)!important;padding:3px!important;border:1px solid var(--bg4)!important;}
.stTabs [data-baseweb="tab"]{color:var(--w2)!important;border-radius:8px!important;}
.stTabs [aria-selected="true"]{background:var(--bg3)!important;color:var(--cyan)!important;}
.stExpander{background:var(--bg2)!important;border:1px solid var(--bg4)!important;
    border-radius:var(--r)!important;}
.stFileUploader{background:var(--bg2)!important;
    border:1px dashed rgba(0,229,255,0.25)!important;border-radius:var(--r2)!important;}
</style>"""

def inject_css():
    import streamlit as st
    st.markdown(MASTER_CSS, unsafe_allow_html=True)

def page_header(title, subtitle=""):
    import streamlit as st
    st.markdown(f"""
    <div style="padding:36px 0 18px;animation:fadeUp 0.6s ease-out">
        <div class="hero-title">{title}</div>
        {'<p style="color:#616161;font-size:0.88rem;margin:8px 0 0">'+subtitle+'</p>' if subtitle else ''}
    </div>""", unsafe_allow_html=True)

def metric_row(cards):
    import streamlit as st
    st.markdown(f'<div class="metric-grid">{"".join(cards)}</div>', unsafe_allow_html=True)

def metric_card(val, label):
    return f'<div class="metric-card"><div class="m-val">{val}</div><div class="m-lbl">{label}</div></div>'

def section_title(text):
    import streamlit as st
    st.markdown(f'<div class="sec-title">{text}</div>', unsafe_allow_html=True)

def divider():
    import streamlit as st
    st.markdown('<div class="div"></div>', unsafe_allow_html=True)

def violation_card(vtype, conf, fine, plate=None, desc=""):
    p = f'<span class="plate" style="font-size:0.7rem;padding:2px 8px;margin-top:4px;display:inline-block">{plate}</span>' if plate else ""
    return f"""<div class="v-card"><div style="flex:1">
        <div class="v-name">&#9656; {vtype}</div>
        <div class="v-desc">{desc}</div>{p}</div>
        <div style="text-align:right">
            <div class="v-fine">Rs.{fine:,}</div>
            <div class="v-conf">{conf:.0%}</div>
        </div></div>"""

def no_violation():
    import streamlit as st
    st.markdown("""<div class="safe-card">
        <div style="font-size:1.4rem;color:#DBE2DC">&#10003;</div>
        <div class="s-text">No Violations Detected</div>
        <div class="s-sub">All traffic rules appear to be followed</div>
    </div>""", unsafe_allow_html=True)

def conf_bar(label, val):
    pct = int(val * 100)
    return f"""<div class="cbar-wrap">
        <div class="cbar-row"><span>{label}</span>
        <span style="color:#DBE2DC">{pct}%</span></div>
        <div class="cbar-track"><div class="cbar-fill" style="width:{pct}%"></div></div>
    </div>"""

def live_badge():
    return '<span class="live"><span class="dot"></span>LIVE</span>'
