import streamlit as st
import numpy as np
import pandas as pd
import sys
import os

try:
    import torch
    TORCH_AVAILABLE = True
except Exception as e:
    import traceback
    print(f"CRITICAL PYTORCH IMPORT ERROR (app): {e}")
    traceback.print_exc()
    TORCH_AVAILABLE = False

# Add the project root to sys.path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import get_all_batsmen, get_batsman_kpis, get_player_cricinfo_link, get_model_registry, get_strike_rate_by_phase, get_dismissals_by_bowler_style, get_historical_context, get_strike_rate_by_bowler_style, get_average_by_bowler_style, get_all_bowlers, get_bowler_kpis, get_bowler_economy_by_phase, get_bowler_average_by_batsman_type, get_bowler_kpis_by_batsman_style, get_player_styles
from src.features import SequencePreprocessor
from src.model import get_model
from src.nlp import CommentaryParser

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

# Fetch dynamic bowler list from backend API
all_bowlers_list = get_all_bowlers()
target_bowler = st.sidebar.selectbox("Target Bowler Profile", all_bowlers_list, index=all_bowlers_list.index("AJ Tye") if "AJ Tye" in all_bowlers_list else 0)



# Main Dashboard Container
st.markdown("<p style='color:#94A3B8; text-align:center;'>Real-time sequence sequence-based analytics for short-format cricket matches.</p>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Batsman Profile", 
    "🎙️ NLP Commentary", 
    "🧠 Live Simulator", 
    "⚙️ ML System",
    "🎯 Bowler Profile"
])

# --- TAB 1: BATSMAN PROFILE ---
with tab1:
    # Get Cricinfo link
    cricinfo_link = get_player_cricinfo_link(selected_batsman)
    link_html = f" <a href='{cricinfo_link}' target='_blank' style='font-size: 1.2rem; text-decoration: none; color: #38BDF8;'>[ESPNCricinfo 🔗]</a>" if cricinfo_link else ""
    st.markdown(f"## 🏏 Batsman Profile: {selected_batsman}{link_html}", unsafe_allow_html=True)

    bat_style, bowl_style = get_player_styles(selected_batsman)
    badges = []
    if bat_style: badges.append(f"<span style='background-color:#E53E3E; color:white; padding: 4px 10px; border-radius: 12px; font-size:0.9rem; font-weight:bold; margin-right:8px;'>🏏 {bat_style}</span>")
    if bowl_style: badges.append(f"<span style='background-color:#3182CE; color:white; padding: 4px 10px; border-radius: 12px; font-size:0.9rem; font-weight:bold;'>🎯 {bowl_style}</span>")
    if badges: st.markdown("".join(badges), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

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

    st.markdown("### Contextual Chart Analytics")
    col_chart1, col_chart2, col_chart3, col_chart4 = st.columns(4)
    with col_chart1:
        st.markdown("#### SR by Match Phase")
        df_phase = get_strike_rate_by_phase(selected_batsman)
        st.bar_chart(df_phase.set_index("Phase"), color="#38BDF8")
    
    with col_chart2:
        st.markdown("#### Dismissals by Style")
        df_style = get_dismissals_by_bowler_style(selected_batsman)
        st.bar_chart(df_style.set_index("Bowler Sub-Style"), color="#2C5282")
        
    with col_chart3:
        st.markdown("#### SR by Bowler Style")
        df_sr_style = get_strike_rate_by_bowler_style(selected_batsman)
        st.bar_chart(df_sr_style.set_index("Bowler Sub-Style"), color="#9333EA")
        
    with col_chart4:
        st.markdown("#### Avg by Bowler Style")
        df_avg_style = get_average_by_bowler_style(selected_batsman)
        st.bar_chart(df_avg_style.set_index("Bowler Sub-Style"), color="#E53E3E")

    # --- ML VULNERABILITY ALERT ---
    try:
        df_vuln = pd.read_csv("data/processed/global_vulnerabilities.csv")
        player_vuln = df_vuln[df_vuln['Player'] == selected_batsman]
        if not player_vuln.empty:
            p_line = player_vuln.iloc[0]['Primary_Weakness_Line']
            p_length = player_vuln.iloc[0]['Primary_Weakness_Length']
            st.error(f"🚨 **MACHINE LEARNING VULNERABILITY DETECTED:** Historical NLP commentary analysis indicates **{selected_batsman}** is highly susceptible to **{p_length.upper()}** deliveries on the **{p_line.upper()}** line.")
    except Exception as e:
        pass

    # --- GLOBAL MATRIX EXPANDER ---
    st.markdown("### 🧠 Machine Learning Insights")
    with st.expander("View Global Player Vulnerability Matrix"):
        st.markdown("This matrix is algorithmically generated by passing historical HuggingFace commentary strings through our custom Scikit-Learn NLP Pipeline.")
        try:
            df_vuln_full = pd.read_csv("data/processed/global_vulnerabilities.csv")
            st.dataframe(df_vuln_full, use_container_width=True)
        except Exception as e:
            st.warning("Global Vulnerability Matrix not found. Run `analyze_players.py` to generate it.")

# --- TAB 2: COMMENTARY NLP PARSER ---
with tab2:
    st.markdown("### Commentary Feature Extraction Playground")
    st.markdown("Paste ball-by-ball commentary text below to test the Machine Learning NLP extraction.")
    
    sample_text = "Starc bowls a full delivery outside off-stump, Kohli attempts a drive but edges it to first slip for a dismissal"
    commentary_input = st.text_area("Ball Commentary String", value=sample_text, height=100)
    
    if st.button("Parse Commentary Features", type="primary"):
        parser = CommentaryParser()
        features = parser.extract_features(commentary_input)
        
        st.success("ML NLP Parsing Complete!")
        
        st.markdown("#### Participants")
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            st.metric("Extracted Bowler", features.get('bowler', 'Unknown'))
        with col_p2:
            st.metric("Extracted Batter", features.get('batter', 'Unknown'))
        with col_p3:
            st.metric("Extracted Outcome", features.get('outcome', 'Unknown'), help="Delivery result")
            
        st.markdown("#### Delivery Mechanics")
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Extracted Line", features['line'], help="ML classification applied for Line")
        with col_m2:
            st.metric("Extracted Length", features['length'], help="ML classification applied for Length")
        with col_m3:
            st.metric("Extracted Shot Intent", features['shot'], help="ML classification applied for Shot Intent")

# --- TAB 3: LIVE SEQUENCE SIMULATOR (PyTorch Integration) ---
with tab3:
    st.markdown(f"### PyTorch LSTM Simulation: {selected_batsman} vs {target_bowler}")
    st.markdown("Build a dynamic sequence of deliveries faced by the batsman to predict the next-ball error probability using our PyTorch LSTM model.")
    
    # Dynamic LSTM sequence length selector
    seq_length = st.slider("LSTM Sequence Window (Number of past deliveries)", min_value=3, max_value=12, value=6)
    
    st.markdown("#### Match Context for Simulation")
    sim_phase = st.selectbox("Current Match Phase", ["Powerplay", "Middle Overs", "Death Overs"])
        
    # Auto-lookup target bowler style
    from src.utils import load_data
    safe_bowler = target_bowler.replace("'", "''")
    b_style_df = load_data(f"SELECT bowling_style FROM players WHERE name = '{safe_bowler}'")
    raw_b_style = b_style_df['bowling_style'].iloc[0] if not b_style_df.empty and pd.notna(b_style_df['bowling_style'].iloc[0]) else "fast"
    
    raw_lower = raw_b_style.lower()
    if 'spin' in raw_lower or 'orthodox' in raw_lower or 'break' in raw_lower:
        if 'leg' in raw_lower or 'orthodox' in raw_lower or 'chinaman' in raw_lower:
            sim_style = "Leg-spin"
        else:
            sim_style = "Off-spin"
    else:
        sim_style = "Pace"
        
    # Bridge Tab 1 (Historical) to Tab 3 (Simulation Context)
    sr, dismissals, balls, _ = get_historical_context(selected_batsman, sim_phase, sim_style)
    
    # Fetch Bowler's Phase Wickets and Economy to show dynamically in the UI
    from src.utils import get_bowler_wickets_by_phase
    phase_df_ui = get_bowler_economy_by_phase(target_bowler)
    phase_row_ui = phase_df_ui[phase_df_ui['Phase'] == sim_phase]
    ui_phase_econ = phase_row_ui['Economy'].iloc[0] if not phase_row_ui.empty else "N/A"
    
    phase_wkt_df_ui = get_bowler_wickets_by_phase(target_bowler)
    phase_wkt_row_ui = phase_wkt_df_ui[phase_wkt_df_ui['Phase'] == sim_phase]
    ui_phase_wkts = phase_wkt_row_ui['Wickets'].iloc[0] if not phase_wkt_row_ui.empty else 0

    if balls > 0:
        st.info(f"**Historical Context:** {selected_batsman} has faced **{balls} balls** in the **{sim_phase}** against **{sim_style}** bowlers, striking at **{sr}** with **{dismissals} dismissals**. Meanwhile, {target_bowler} has an economy of **{ui_phase_econ}** and has taken **{ui_phase_wkts} wickets** in the **{sim_phase}**.")
    else:
        st.warning(f"**Historical Context:** No historical data found for {selected_batsman} against {sim_style} in the {sim_phase}. {target_bowler} has an economy of **{ui_phase_econ}** and **{ui_phase_wkts} wickets** in this phase.")
    
    st.markdown(f"#### Rolling Sequence Inputs (Last {seq_length} Deliveries)")
    
    # Generate dynamic columns
    cols = st.columns(seq_length)
    sequence_data = []
    
    for i in range(seq_length):
        with cols[i]:
            run = st.selectbox(f"Ball {i+1} Run", [0, 1, 2, 4, 6], key=f"d_run_{i}")
            length = st.selectbox(f"Ball {i+1} Length", ["Yorker", "Full", "Slot", "Good Length", "Short"], key=f"d_len_{i}")
            sequence_data.append({'run': run, 'length': length})

    if st.button("Predict PyTorch Vulnerability", type="primary"):
        # Fetch Batsman's Style
        from src.utils import load_data
        safe_bat = selected_batsman.replace("'", "''")
        bat_df = load_data(f"SELECT batting_style FROM players WHERE name = '{safe_bat}'")
        bat_style = bat_df['batting_style'].iloc[0] if not bat_df.empty and pd.notna(bat_df['batting_style'].iloc[0]) else "Right-hand bat"
        
        # Fetch historical contextual KPIs
        sr, dismissals, balls_faced, batsman_avg_vs_style = get_historical_context(selected_batsman, sim_phase, sim_style)
        
        # Fetch Bowler's Phase Economy & Wickets
        phase_df = get_bowler_economy_by_phase(target_bowler)
        phase_row = phase_df[phase_df['Phase'] == sim_phase]
        b_phase_econ = phase_row['Economy'].iloc[0] if not phase_row.empty else 7.5
        
        phase_wkts_df = get_bowler_wickets_by_phase(target_bowler)
        phase_wkts_row = phase_wkts_df[phase_wkts_df['Phase'] == sim_phase]
        b_phase_wkts = phase_wkts_row['Wickets'].iloc[0] if not phase_wkts_row.empty else 0
        
        # Fetch Bowler's Average against Batsman Type
        avg_df = get_bowler_average_by_batsman_type(target_bowler)
        avg_row = avg_df[avg_df['Batsman Type'] == bat_style]
        b_type_avg = avg_row['Average'].iloc[0] if not avg_row.empty else 25.0
        
        # New: Fetch bowler's deep specific KPIs against this batsman's style
        b_econ_vs_style, b_sr_vs_style, b_wkts_vs_style = get_bowler_kpis_by_batsman_style(target_bowler, bat_style)
        
        # Fetch Bowler's Career KPIs
        b_wkts, b_runs, b_career_econ, b_career_avg, b_career_sr = get_bowler_kpis(target_bowler)
        if b_career_avg == "N/A":
            b_career_avg = 25.0
            
        # 1. Preprocess the sequence into a 24-dimensional tensor using historical context
        preprocessor = SequencePreprocessor()
        input_tensor = preprocessor.preprocess_sequence(
            sequence_data=sequence_data,
            phase=sim_phase,
            style=sim_style,
            norm_sr=sr / 200.0,
            dismissal_rate=dismissals / balls_faced if balls_faced > 0 else 0,
            bowler_phase_econ=b_phase_econ,
            bowler_phase_wickets=b_phase_wkts,
            bowler_type_avg=b_type_avg,
            bowler_career_wickets=b_wkts,
            bowler_career_econ=b_career_econ,
            bowler_career_avg=b_career_avg,
            bowler_career_sr=b_career_sr,
            batsman_avg_vs_style=batsman_avg_vs_style,
            bowler_econ_vs_style=b_econ_vs_style,
            bowler_sr_vs_style=b_sr_vs_style,
            bowler_wkts_vs_style=b_wkts_vs_style
        )
        
        # 2. Load the LSTM model
        model = get_model()
        
        # 3. Run inference (PyTorch or NumPy fallback)
        if TORCH_AVAILABLE:
            with torch.no_grad():
                prediction_tensor = model(input_tensor)
                risk_score = prediction_tensor.item()
        else:
            prediction_tensor = model(input_tensor)
            risk_score = prediction_tensor.item()
        
        st.markdown("#### Neural Network Sequence Prediction")
        st.progress(float(min(risk_score, 1.0)))
        
        st.metric("LSTM Vulnerability Risk Score", f"{risk_score * 100:.2f}%", 
                  help="Probability output directly from the PyTorch Sigmoid layer.")
        
        with st.expander("View Raw PyTorch Tensor Output"):
            st.code(f"Input Shape: {input_tensor.shape}\nOutput Tensor: {prediction_tensor}\nItem Value: {risk_score}")
            
    st.divider()
    st.markdown("#### ⚙️ Week 5 ML Model Registry (SQLite Backend)")
    st.markdown("This table dynamically pulls from the `model_registry` table in our database. It proves that the PyTorch training pipeline (`train.py`) successfully logged its focal loss and weight filepaths!")
    registry_df = get_model_registry()
    if not registry_df.empty:
        st.dataframe(registry_df, width='stretch', hide_index=True)
    else:
        st.info("No models found in the database. Run `python src/train.py` in the terminal to train and log a model!")


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

# --- TAB 5: BOWLER PROFILE ---
with tab5:
    cricinfo_link = get_player_cricinfo_link(target_bowler)
    link_html = f" <a href='{cricinfo_link}' target='_blank' style='font-size: 1.2rem; text-decoration: none; color: #38BDF8;'>[ESPNCricinfo 🔗]</a>" if cricinfo_link else ""
    st.markdown(f"## 🎯 Bowler Profile: {target_bowler}{link_html}", unsafe_allow_html=True)
    
    bat_style, bowl_style = get_player_styles(target_bowler)
    badges = []
    if bowl_style: badges.append(f"<span style='background-color:#3182CE; color:white; padding: 4px 10px; border-radius: 12px; font-size:0.9rem; font-weight:bold; margin-right:8px;'>🎯 {bowl_style}</span>")
    if bat_style: badges.append(f"<span style='background-color:#E53E3E; color:white; padding: 4px 10px; border-radius: 12px; font-size:0.9rem; font-weight:bold;'>🏏 {bat_style}</span>")
    if badges: st.markdown("".join(badges), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    wickets, runs_conc, economy, avg, sr = get_bowler_kpis(target_bowler)
    
    st.markdown("### Career T20I Statistics")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{wickets}</div><div class="kpi-label">Wickets</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{economy}</div><div class="kpi-label">Economy Rate</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{avg}</div><div class="kpi-label">Bowling Average</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{sr}</div><div class="kpi-label">Strike Rate</div></div>', unsafe_allow_html=True)
    with col5:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{runs_conc}</div><div class="kpi-label">Runs Conceded</div></div>', unsafe_allow_html=True)
        
    st.markdown("### Contextual Chart Analytics")
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.markdown("#### Economy by Match Phase")
        from src.utils import get_bowler_economy_by_phase
        df_phase_econ = get_bowler_economy_by_phase(target_bowler)
        st.bar_chart(df_phase_econ.set_index("Phase")['Economy'])
    
    with col_chart2:
        st.markdown("#### Wickets by Match Phase")
        from src.utils import get_bowler_wickets_by_phase
        df_phase_wkts = get_bowler_wickets_by_phase(target_bowler)
        st.bar_chart(df_phase_wkts.set_index("Phase")['Wickets'], color="#48BB78")
        
    df_bat_type = get_bowler_average_by_batsman_type(target_bowler)
    
    col_chart1, col_chart2, col_chart3, col_chart4 = st.columns(4)
    with col_chart1:
        st.markdown("#### Average by Batsman Type")
        st.bar_chart(df_bat_type.set_index("Batsman Type")['Average'], color="#F56565")
        
    with col_chart2:
        st.markdown("#### Economy by Batsman Type")
        st.bar_chart(df_bat_type.set_index("Batsman Type")['Economy'], color="#38A169")
        
    with col_chart3:
        st.markdown("#### Strike Rate by Batsman Type")
        st.bar_chart(df_bat_type.set_index("Batsman Type")['Strike Rate'], color="#805AD5")
        
    with col_chart4:
        st.markdown("#### Wickets by Batsman Type")
        st.bar_chart(df_bat_type.set_index("Batsman Type")['Wickets'], color="#3182CE")
        
    # --- ML STRENGTH ALERT ---
    try:
        df_strength = pd.read_csv("data/processed/global_bowler_strengths.csv")
        player_str = df_strength[df_strength['Player'] == target_bowler]
        if not player_str.empty:
            p_line = player_str.iloc[0]['Primary_Strength_Line']
            p_length = player_str.iloc[0]['Primary_Strength_Length']
            st.success(f"🎯 **MACHINE LEARNING LETHALITY DETECTED:** Historical NLP commentary analysis indicates **{target_bowler}** is most likely to take wickets with **{p_length.upper()}** deliveries on the **{p_line.upper()}** line.")
    except Exception as e:
        pass
