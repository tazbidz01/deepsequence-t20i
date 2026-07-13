import streamlit as st
import numpy as np
import pandas as pd
from src.utils import get_all_batsmen, get_batsman_kpis, get_player_cricinfo_link

# Set page config
st.set_page_config(
    page_title="DeepSequence-T20I: Strategic Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling for UI wow-factor
st.markdown("""
<style>
    .main {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: #1E293B;
        padding: 8px 16px;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #94A3B8;
        font-weight: 600;
        border-bottom: none;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #38BDF8;
    }
    .stTabs [aria-selected="true"] {
        color: #38BDF8 !important;
        border-bottom: 2px solid #38BDF8 !important;
    }
    .kpi-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #38BDF8;
        margin-bottom: 4px;
    }
    .kpi-label {
        font-size: 0.85rem;
        font-weight: 500;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Controller
st.sidebar.markdown("<h2 style='color:#38BDF8; margin-bottom:0;'>DeepSequence-T20I</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color:#64748B; font-size:0.85rem; margin-top:0;'>Contextual Batsman Vulnerability Engine</p>", unsafe_allow_html=True)
st.sidebar.divider()

# Fetch dynamic batsman list from backend API
batsman_list = get_all_batsmen()
default_idx = batsman_list.index("AJ Finch") if "AJ Finch" in batsman_list else 0
selected_batsman = st.sidebar.selectbox("Target Batsman Profile", batsman_list, index=default_idx)

st.sidebar.markdown("### Match Context Filters")
bowler_hand = st.sidebar.radio("Bowler Delivery Hand", ["All", "Right-Arm Only", "Left-Arm Only"])
match_phase = st.sidebar.multiselect("Match Phase Segment", ["Powerplay (0-6)", "Middle Overs (7-15)", "Death Overs (16-20)"], default=["Powerplay (0-6)", "Middle Overs (7-15)", "Death Overs (16-20)"])

st.sidebar.divider()
st.sidebar.caption("CSE299.13 Junior Design Project proposal UI Wireframe skeleton.")

# Get Cricinfo link
cricinfo_link = get_player_cricinfo_link(selected_batsman)
link_html = f" <a href='{cricinfo_link}' target='_blank' style='font-size: 1.2rem; text-decoration: none; color: #38BDF8;'>[ESPNCricinfo 🔗]</a>" if cricinfo_link else ""

# Main Dashboard Container
st.markdown(f"# Tactical Analysis Profile: {selected_batsman}{link_html}", unsafe_allow_html=True)
st.markdown("<p style='color:#94A3B8;'>Real-time sequence sequence-based analytics for short-format cricket matches.</p>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Batsman Profile Dashboard",
    "📝 Commentary NLP Parser",
    "🔄 Live Sequence Simulator",
    "🛡️ Strategic Plan-of-Attack"
])

# --- TAB 1: BATSMAN PROFILE ---
with tab1:
    st.markdown("### Contextual Metrics Aggregates")
    
    # Fetch real stats via backend utils
    total_runs, balls_faced, strike_rate, times_out = get_batsman_kpis(selected_batsman)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{total_runs}</div><div class="kpi-label">Total Runs</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{balls_faced}</div><div class="kpi-label">Balls Faced</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{strike_rate}</div><div class="kpi-label">Strike Rate</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{times_out}</div><div class="kpi-label">Times Out</div></div>', unsafe_allow_html=True)

    st.markdown("### Match Phase & Bowling Splits")
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.markdown("#### Strike Rate by Match Phase")
        # Mock chart for now until we build Phase logic
        df_phase = pd.DataFrame({
            "Phase": ["Powerplay (0-6)", "Middle (7-15)", "Death (16-20)"],
            "Strike Rate": [128.5, 138.2, 172.4]
        })
        st.bar_chart(df_phase.set_index("Phase"), color="#38BDF8")
    
    with col_chart2:
        st.markdown("#### Dismissal Splits by Bowler Style")
        df_style = pd.DataFrame({
            "Bowler Sub-Style": ["Pace (Fast)", "Off-spin", "Leg-spin", "Chinaman"],
            "Dismissals": [18, 5, 9, 2]
        })
        st.bar_chart(df_style.set_index("Bowler Sub-Style"), color="#2C5282")

# --- TAB 2: COMMENTARY NLP PARSER ---
with tab2:
    st.markdown("### Commentary Feature Extraction Playground")
    st.markdown("Paste ball-by-ball commentary text below to test regex extraction patterns.")
    
    sample_text = "Starc bowls a full delivery outside off-stump, Kohli attempts a drive but edges it to first slip for a dismissal"
    commentary_input = st.text_area("Ball Commentary String", value=sample_text, height=100)
    
    if st.button("Parse Commentary Features", type="primary"):
        st.success("Regex Parsing Complete!")
        col_res1, col_res2, col_res3 = st.columns(3)
        with col_res1:
            st.metric("Extracted Line", "Outside Off", help="Regex match: 'outside off'")
        with col_res2:
            st.metric("Extracted Length", "Full", help="Regex match: 'full delivery'")
        with col_res3:
            st.metric("Extracted Shot Intent", "Drive", help="Regex match: 'attempts a drive'")

# --- TAB 3: LIVE SEQUENCE SIMULATOR ---
with tab3:
    st.markdown("### Rolling Sequence Vulnerability Simulator")
    st.markdown("Build a sequence of deliveries faced by the batsman to predict the next-ball error probability using the LSTM model.")
    
    st.markdown("#### Rolling Sequence Inputs (Last 6 Deliveries)")
    col_s1, col_s2, col_s3, col_s4, col_s5, col_s6 = st.columns(6)
    with col_s1:
        st.selectbox("Ball 1 Run", [0, 1, 2, 4, 6], key="s1")
        st.selectbox("Ball 1 Length", ["Yorker", "Full", "Slot", "Good Length", "Short"], key="s1_len")
    with col_s2:
        st.selectbox("Ball 2 Run", [0, 1, 2, 4, 6], key="s2")
        st.selectbox("Ball 2 Length", ["Yorker", "Full", "Slot", "Good Length", "Short"], key="s2_len")
    with col_s3:
        st.selectbox("Ball 3 Run", [0, 1, 2, 4, 6], key="s3")
        st.selectbox("Ball 3 Length", ["Yorker", "Full", "Slot", "Good Length", "Short"], key="s3_len")
    with col_s4:
        st.selectbox("Ball 4 Run", [0, 1, 2, 4, 6], index=1, key="s4")
        st.selectbox("Ball 4 Length", ["Yorker", "Full", "Slot", "Good Length", "Short"], key="s4_len")
    with col_s5:
        st.selectbox("Ball 5 Run", [0, 1, 2, 4, 6], index=1, key="s5")
        st.selectbox("Ball 5 Length", ["Yorker", "Full", "Slot", "Good Length", "Short"], key="s5_len")
    with col_s6:
        st.selectbox("Ball 6 Run", [0, 1, 2, 4, 6], index=3, key="s6")
        st.selectbox("Ball 6 Length", ["Yorker", "Full", "Slot", "Good Length", "Short"], key="s6_len")

    if st.button("Predict Next-Ball Vulnerability", type="primary"):
        runs = [st.session_state.get(f"s{i}") for i in range(1, 7)]
        dot_count = sum(1 for r in runs if r == 0)
        base_risk = 0.05 + (dot_count * 0.12)
        
        st.markdown("#### Sequence Prediction Result")
        st.progress(float(min(base_risk, 1.0)))
        st.metric("Vulnerability Risk Score", f"{min(base_risk * 100, 100.0):.1f}%", help="Probability of batsman getting out or committing an error on the upcoming ball.")

# --- TAB 4: STRATEGIC PLAN-OF-ATTACK ---
with tab4:
    st.markdown("### Tactical Cheat Sheets Generator")
    st.markdown("Generate and compile strategic Plan-of-Attack data sheets designed for opponent profiles.")
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.selectbox("Target Bowler Type to Generate Strategy", ["Right-Arm Fast (Pace)", "Left-Arm Fast (Pace)", "Off-Spin (Right-Arm)", "Leg-Spin (Right-Arm)"])
        st.write("---")
        st.markdown("**Vulnerability Vector Summary:**")
        st.info("Opponent profiles reveal a sequence-based vulnerability when faced with consecutive good-length deliveries on middle-off line, followed by a wide yorker.")
    
    with col_p2:
        st.markdown("#### PDF Strategy Export")
        st.write("Click below to compile and download the official PDF cheat sheet containing visual strategy guidelines.")
        
        st.button("Compile & Download PDF Report", type="secondary")
