import pandas as pd
import streamlit as st
from src.db import get_connection

def load_data(query):
    try:
        conn = get_connection()
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"Database error: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_all_batsmen():
    df = load_data("SELECT DISTINCT batter FROM deliveries ORDER BY batter ASC")
    if not df.empty:
        return df['batter'].tolist()
    return ["No Data Available"]

@st.cache_data(ttl=3600)
def get_batsman_kpis(batter_name):
    # Escape quotes in names like D'Arcy Short
    safe_name = batter_name.replace("'", "''")
    query = f"""
        SELECT 
            SUM(runs_batter) as total_runs, 
            COUNT(*) as balls_faced, 
            SUM(CASE WHEN player_out = '{safe_name}' THEN 1 ELSE 0 END) as times_out 
        FROM deliveries 
        WHERE batter = '{safe_name}'
    """
    df = load_data(query)
    
    if not df.empty and df.iloc[0]['balls_faced'] > 0:
        total_runs = int(df.iloc[0]['total_runs'] or 0)
        balls_faced = int(df.iloc[0]['balls_faced'] or 0)
        times_out = int(df.iloc[0]['times_out'] or 0)
        strike_rate = round((total_runs / balls_faced) * 100, 2) if balls_faced > 0 else 0
        return total_runs, balls_faced, strike_rate, times_out
    
    return 0, 0, 0, 0

@st.cache_data(ttl=3600)
def get_player_cricinfo_link(player_name):
    safe_name = player_name.replace("'", "''")
    query = f"SELECT cricinfo_id FROM players WHERE name = '{safe_name}' LIMIT 1"
    df = load_data(query)
    
    if not df.empty and pd.notna(df.iloc[0]['cricinfo_id']):
        cricinfo_id = df.iloc[0]['cricinfo_id']
        return f"https://www.espncricinfo.com/player/player-{int(float(cricinfo_id))}"
    return None

@st.cache_data(ttl=3600)
def get_player_styles(player_name):
    safe_name = player_name.replace("'", "''")
    query = f"SELECT batting_style, bowling_style FROM players WHERE name = '{safe_name}' LIMIT 1"
    df = load_data(query)
    
    if not df.empty:
        bat_style = df.iloc[0]['batting_style'] if pd.notna(df.iloc[0]['batting_style']) else ""
        bowl_style = df.iloc[0]['bowling_style'] if pd.notna(df.iloc[0]['bowling_style']) else ""
        return bat_style, bowl_style
    return "", ""

def log_model_training(version, loss, filepath):
    try:
        from datetime import datetime
        date_trained = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO model_registry (version, loss, filepath, date_trained)
            VALUES (?, ?, ?, ?)
        ''', (version, loss, filepath, date_trained))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Failed to log model training: {e}")
        return False

def get_model_registry():
    query = "SELECT version, loss as focal_loss, filepath, date_trained FROM model_registry ORDER BY date_trained DESC"
    return load_data(query)

@st.cache_data(ttl=3600)
def get_strike_rate_by_phase(batter_name):
    safe_name = batter_name.replace("'", "''")
    query = f"""
        SELECT 
            CASE 
                WHEN over_num BETWEEN 0 AND 5 THEN 'Powerplay (0-5)'
                WHEN over_num BETWEEN 6 AND 14 THEN 'Middle (6-14)'
                WHEN over_num BETWEEN 15 AND 19 THEN 'Death (15-19)'
            END as Phase,
            CASE 
                WHEN over_num BETWEEN 0 AND 5 THEN 1
                WHEN over_num BETWEEN 6 AND 14 THEN 2
                WHEN over_num BETWEEN 15 AND 19 THEN 3
            END as phase_order,
            SUM(runs_batter) as total_runs,
            COUNT(*) as balls_faced
        FROM deliveries
        WHERE batter = '{safe_name}'
        GROUP BY Phase, phase_order
        ORDER BY phase_order
    """
    df = load_data(query)
    
    if df.empty:
        return pd.DataFrame({"Phase": ["Powerplay (0-5)", "Middle (6-14)", "Death (15-19)"], "Strike Rate": [0.0, 0.0, 0.0]})
    
    df['Strike Rate'] = df.apply(lambda row: round((row['total_runs'] / row['balls_faced']) * 100, 2) if row['balls_faced'] > 0 else 0.0, axis=1)
    
    return df[['Phase', 'Strike Rate']]

@st.cache_data(ttl=3600)
def get_dismissals_by_bowler_style(batter_name):
    safe_name = batter_name.replace("'", "''")
    
    # Fetch real bowlers who dismissed this batsman and JOIN to get their true scraped style
    query = f"""
        SELECT p.bowling_style as "Bowler Sub-Style", COUNT(*) as dismissals
        FROM deliveries d
        JOIN players p ON d.bowler = p.name
        WHERE d.player_out = '{safe_name}' AND p.bowling_style IS NOT NULL AND p.bowling_style != ''
        GROUP BY p.bowling_style
        ORDER BY dismissals DESC
    """
    df = load_data(query)
    
    if df.empty:
        return pd.DataFrame({"Bowler Sub-Style": ["No Scraped Data"], "Dismissals": [0]})
    
    return df

@st.cache_data(ttl=3600)
def get_strike_rate_by_bowler_style(batter_name):
    safe_name = batter_name.replace("'", "''")
    query = f"""
        SELECT 
            p.bowling_style as "Bowler Sub-Style",
            SUM(d.runs_batter) as total_runs,
            COUNT(*) as balls_faced
        FROM deliveries d
        JOIN players p ON d.bowler = p.name
        WHERE d.batter = '{safe_name}' AND p.bowling_style IS NOT NULL AND p.bowling_style != ''
        GROUP BY p.bowling_style
        HAVING balls_faced > 0
        ORDER BY total_runs DESC
    """
    df = load_data(query)
    
    if df.empty:
        return pd.DataFrame({"Bowler Sub-Style": ["No Data"], "Strike Rate": [0.0]})
    
    df['Strike Rate'] = df.apply(lambda row: round((row['total_runs'] / row['balls_faced']) * 100, 2), axis=1)
    
    return df[['Bowler Sub-Style', 'Strike Rate']]

@st.cache_data(ttl=3600)
def get_average_by_bowler_style(batter_name):
    safe_name = batter_name.replace("'", "''")
    query = f"""
        SELECT 
            p.bowling_style as "Bowler Sub-Style",
            SUM(d.runs_batter) as total_runs,
            SUM(CASE WHEN d.player_out = '{safe_name}' THEN 1 ELSE 0 END) as dismissals
        FROM deliveries d
        JOIN players p ON d.bowler = p.name
        WHERE d.batter = '{safe_name}' AND p.bowling_style IS NOT NULL AND p.bowling_style != ''
        GROUP BY p.bowling_style
        HAVING total_runs > 0
        ORDER BY total_runs DESC
    """
    df = load_data(query)
    
    if df.empty:
        return pd.DataFrame({"Bowler Sub-Style": ["No Data"], "Average": [0.0]})
        
    df['Average'] = df.apply(lambda row: round(row['total_runs'] / row['dismissals'], 2) if row['dismissals'] > 0 else row['total_runs'], axis=1)
    
    return df[['Bowler Sub-Style', 'Average']]

@st.cache_data(ttl=3600)
def get_partnership_stats(player1, player2):
    """
    Returns the raw historical partnership stats: (runs, balls, dismissals).
    """
    safe_p1 = player1.replace("'", "''")
    safe_p2 = player2.replace("'", "''")
    query = f"""
        SELECT
            SUM(runs_batter + runs_extras) as total_runs,
            COUNT(*) as total_balls,
            COUNT(CASE WHEN player_out = '{safe_p1}' OR player_out = '{safe_p2}' THEN 1 END) as dismissals,
            COUNT(DISTINCT match_id) as total_matches
        FROM deliveries
        WHERE (batter = '{safe_p1}' AND non_striker = '{safe_p2}')
           OR (batter = '{safe_p2}' AND non_striker = '{safe_p1}')
    """
    df = load_data(query)
    
    if df.empty or pd.isna(df['total_balls'].iloc[0]) or df['total_balls'].iloc[0] == 0:
        return 0, 0, 0, 0
        
    runs = int(df['total_runs'].iloc[0])
    balls = int(df['total_balls'].iloc[0])
    outs = int(df['dismissals'].iloc[0])
    matches = int(df['total_matches'].iloc[0])
    
    return runs, balls, outs, matches

@st.cache_data(ttl=3600)
def get_partnership_strength(player1, player2):
    """
    Returns a normalized scalar (0.0 to 1.0) representing how strong player1 and player2 bat together.
    Based on their historical run rate and average partnership runs.
    """
    safe_p1 = player1.replace("'", "''")
    safe_p2 = player2.replace("'", "''")
    query = f"""
        SELECT
            SUM(runs_batter + runs_extras) as total_runs,
            COUNT(*) as total_balls,
            COUNT(CASE WHEN player_out = '{safe_p1}' OR player_out = '{safe_p2}' THEN 1 END) as dismissals
        FROM deliveries
        WHERE (batter = '{safe_p1}' AND non_striker = '{safe_p2}')
           OR (batter = '{safe_p2}' AND non_striker = '{safe_p1}')
    """
    df = load_data(query)
    
    if df.empty or pd.isna(df['total_balls'].iloc[0]) or df['total_balls'].iloc[0] == 0:
        return 0.5 # Default neutral strength if no history
        
    runs = float(df['total_runs'].iloc[0])
    balls = float(df['total_balls'].iloc[0])
    outs = float(df['dismissals'].iloc[0])
    
    # Partnership SR (Scale: 120 SR = neutral, higher is better)
    sr = (runs / balls) * 100.0 if balls > 0 else 0
    # Average partnership runs before getting out
    avg = runs / outs if outs > 0 else runs
    
    # Normalize
    norm_sr = min(max((sr - 80) / 100.0, 0.0), 1.0) # 80SR->0, 180SR->1
    norm_avg = min(max((avg - 10) / 60.0, 0.0), 1.0) # 10 avg->0, 70 avg->1
    
    return (norm_sr * 0.5) + (norm_avg * 0.5)

# ==========================================
# BOWLER PROFILE ANALYTICS (TAB 5)
# ==========================================

@st.cache_data(ttl=3600)
def get_all_bowlers():
    query = "SELECT DISTINCT bowler FROM deliveries ORDER BY bowler"
    df = load_data(query)
    return df['bowler'].tolist()

@st.cache_data(ttl=3600)
def get_bowler_kpis(bowler_name):
    safe_name = bowler_name.replace("'", "''")
    query = f"""
        SELECT 
            COUNT(CASE WHEN wicket_type != '' AND wicket_type != 'run out' THEN 1 END) as wickets,
            SUM(runs_batter + runs_extras) as runs_conceded,
            COUNT(*) as balls_bowled
        FROM deliveries 
        WHERE bowler = '{safe_name}'
    """
    df = load_data(query)
    if df.empty or pd.isna(df['balls_bowled'].iloc[0]) or df['balls_bowled'].iloc[0] == 0:
        return 0, 0, 0, 0, 0
    
    wickets = int(df['wickets'].iloc[0])
    runs = int(df['runs_conceded'].iloc[0])
    balls = int(df['balls_bowled'].iloc[0])
    overs = balls / 6.0
    
    economy = round(runs / overs, 2) if overs > 0 else 0
    average = round(runs / wickets, 2) if wickets > 0 else "N/A"
    sr = round(balls / wickets, 2) if wickets > 0 else "N/A"
    
    return wickets, runs, economy, average, sr

@st.cache_data(ttl=3600)
def get_bowler_economy_by_phase(bowler_name):
    safe_name = bowler_name.replace("'", "''")
    query = f"""
        SELECT 
            CASE 
                WHEN over_num < 6 THEN 'Powerplay'
                WHEN over_num >= 6 AND over_num < 15 THEN 'Middle Overs'
                ELSE 'Death Overs'
            END as Phase,
            SUM(runs_batter + runs_extras) as total_runs,
            COUNT(*) as total_balls
        FROM deliveries
        WHERE bowler = '{safe_name}'
        GROUP BY Phase
        ORDER BY 
            CASE Phase
                WHEN 'Powerplay' THEN 1
                WHEN 'Middle Overs' THEN 2
                WHEN 'Death Overs' THEN 3
            END
    """
    df = load_data(query)
    if df.empty:
        return pd.DataFrame({"Phase": ["Powerplay", "Middle Overs", "Death Overs"], "Economy": [0.0, 0.0, 0.0]})
        
    df['Economy'] = df.apply(lambda row: round(row['total_runs'] / (row['total_balls'] / 6.0), 2) if row['total_balls'] > 0 else 0, axis=1)
    return df[['Phase', 'Economy']]

@st.cache_data(ttl=3600)
def get_bowler_wickets_by_phase(bowler_name):
    safe_name = bowler_name.replace("'", "''")
    query = f"""
        SELECT 
            CASE 
                WHEN over_num < 6 THEN 'Powerplay'
                WHEN over_num >= 6 AND over_num < 15 THEN 'Middle Overs'
                ELSE 'Death Overs'
            END as Phase,
            COUNT(CASE WHEN wicket_type != '' AND wicket_type != 'run out' THEN 1 END) as Wickets
        FROM deliveries
        WHERE bowler = '{safe_name}'
        GROUP BY Phase
        ORDER BY 
            CASE Phase
                WHEN 'Powerplay' THEN 1
                WHEN 'Middle Overs' THEN 2
                WHEN 'Death Overs' THEN 3
            END
    """
    df = load_data(query)
    if df.empty:
        return pd.DataFrame({"Phase": ["Powerplay", "Middle Overs", "Death Overs"], "Wickets": [0, 0, 0]})
    return df[['Phase', 'Wickets']]

@st.cache_data(ttl=3600)
def get_bowler_average_by_batsman_type(bowler_name):
    safe_name = bowler_name.replace("'", "''")
    query = f"""
        SELECT 
            p.batting_style as "Batsman Type",
            SUM(d.runs_batter + d.runs_extras) as total_runs,
            COUNT(CASE WHEN d.wicket_type != '' AND d.wicket_type != 'run out' THEN 1 END) as wickets,
            COUNT(*) as balls_bowled
        FROM deliveries d
        JOIN players p ON d.batter = p.name
        WHERE d.bowler = '{safe_name}' 
          AND p.batting_style IS NOT NULL 
          AND p.batting_style != ''
        GROUP BY "Batsman Type"
        HAVING total_runs > 0

    """
    df = load_data(query)
    if df.empty:
        return pd.DataFrame({"Batsman Type": ["No Data"], "Average": [0.0], "Economy": [0.0], "Strike Rate": [0.0], "Wickets": [0]})
        
    df['Average'] = df.apply(
        lambda row: round(row['total_runs'] / row['wickets'], 2) if row['wickets'] > 0 else row['total_runs'], 
        axis=1
    )
    df['Economy'] = df.apply(
        lambda row: round(row['total_runs'] / (row['balls_bowled'] / 6.0), 2) if row['balls_bowled'] > 0 else 0.0,
        axis=1
    )
    df['Strike Rate'] = df.apply(
        lambda row: round(row['balls_bowled'] / row['wickets'], 2) if row['wickets'] > 0 else float(row['balls_bowled']),
        axis=1
    )
    df['Wickets'] = df['wickets']
    
    return df[['Batsman Type', 'Average', 'Economy', 'Strike Rate', 'Wickets']]

@st.cache_data(ttl=3600)
def get_bowler_kpis_by_batsman_style(bowler_name, batsman_style):
    safe_name = bowler_name.replace("'", "''")
    safe_style = batsman_style.replace("'", "''")
    query = f"""
        SELECT 
            SUM(d.runs_batter + d.runs_extras) as runs_conceded,
            COUNT(CASE WHEN d.wicket_type != '' AND d.wicket_type != 'run out' THEN 1 END) as wickets,
            COUNT(*) as balls_bowled
        FROM deliveries d
        JOIN players p ON d.batter = p.name
        WHERE d.bowler = '{safe_name}'
          AND p.batting_style = '{safe_style}'
    """
    df = load_data(query)
    if df.empty or pd.isna(df['balls_bowled'].iloc[0]) or df['balls_bowled'].iloc[0] == 0:
        return 0.0, 0.0, 0
        
    runs = int(df['runs_conceded'].iloc[0] or 0)
    wickets = int(df['wickets'].iloc[0] or 0)
    balls = int(df['balls_bowled'].iloc[0] or 0)
    
    overs = balls / 6.0
    economy = round(runs / overs, 2) if overs > 0 else 0.0
    sr = round(balls / wickets, 2) if wickets > 0 else float(balls)
    
    return economy, sr, wickets

@st.cache_data(ttl=3600)
def get_historical_context(batter_name, phase_name, style_name):
    safe_name = batter_name.replace("'", "''")
    
    # Map phase to over range
    phase_condition = "1=1"
    if "Powerplay" in phase_name:
        phase_condition = "d.over_num BETWEEN 0 AND 5"
    elif "Middle" in phase_name:
        phase_condition = "d.over_num BETWEEN 6 AND 14"
    elif "Death" in phase_name:
        phase_condition = "d.over_num BETWEEN 15 AND 19"
        
    # Map style macro to SQL likes
    style_condition = "1=1"
    if style_name == "Pace":
        style_condition = "(LOWER(p.bowling_style) LIKE '%fast%' OR LOWER(p.bowling_style) LIKE '%medium%' OR LOWER(p.bowling_style) LIKE '%pace%')"
    elif style_name == "Off-spin":
        style_condition = "(LOWER(p.bowling_style) LIKE '%offbreak%' OR LOWER(p.bowling_style) LIKE '%off spin%')"
    elif style_name == "Leg-spin":
        style_condition = "(LOWER(p.bowling_style) LIKE '%legbreak%' OR LOWER(p.bowling_style) LIKE '%leg spin%' OR LOWER(p.bowling_style) LIKE '%orthodox%')"
        
    query = f"""
        SELECT 
            SUM(d.runs_batter) as runs,
            COUNT(*) as balls,
            SUM(CASE WHEN d.player_out = '{safe_name}' THEN 1 ELSE 0 END) as dismissals
        FROM deliveries d
        JOIN players p ON d.bowler = p.name
        WHERE d.batter = '{safe_name}' 
          AND {phase_condition} 
          AND {style_condition}
    """
    df = load_data(query)
    if not df.empty and df.iloc[0]['balls'] > 0:
        balls = int(df.iloc[0]['balls'])
        runs = int(df.iloc[0]['runs'] or 0)
        dismissals = int(df.iloc[0]['dismissals'] or 0)
        sr = round((runs / balls) * 100, 2)
        avg = round((runs / dismissals), 2) if dismissals > 0 else float(runs)
        return sr, dismissals, balls, avg
    return 0.0, 0, 0, 0.0

@st.cache_data(ttl=3600)
def get_player_specific_vulnerability(batsman_name):
    """
    Dynamically adjusts the baseline vulnerability of Line, Length, and Shot
    based on the specific historical profile of the batsman.
    """
    from src.config import LINE_VULN_SCORES, LENGTH_VULN_SCORES, SHOT_VULN_SCORES
    
    # Get player's career KPIs
    kpis = get_batsman_kpis(batsman_name)
    
    # Create player-specific copies
    p_line = dict(LINE_VULN_SCORES)
    p_len = dict(LENGTH_VULN_SCORES)
    p_shot = dict(SHOT_VULN_SCORES)
    
    if not kpis or 'strike_rate' not in kpis:
        return p_line, p_len, p_shot
        
    sr = float(kpis['strike_rate'])
    runs = int(kpis['runs'])
    
    # 1. Adjust based on Career Strike Rate
    # High SR = lower baseline vulnerability to standard balls
    if sr > 140.0:
        p_len['Good Length'] = max(0.1, p_len['Good Length'] - 0.2)
        p_shot['Drive'] = max(0.1, p_shot['Drive'] - 0.2)
        p_shot['Loft'] = min(0.9, p_shot.get('Loft', 0.9) - 0.1)
    elif sr < 120.0:
        # Struggling players are highly vulnerable to extreme lengths
        p_len['Yorker'] = min(1.0, p_len['Yorker'] + 0.1)
        p_len['Short'] = min(1.0, p_len['Short'] + 0.15)
        p_line['Outside Off'] = min(1.0, p_line['Outside Off'] + 0.15)
        
    # 2. Volume Experience Adjustment
    if runs > 1500:
        # Veterans are less vulnerable to pressure shots
        p_shot['Sweep'] = max(0.1, p_shot['Sweep'] - 0.1)
        p_shot['Pull'] = max(0.1, p_shot['Pull'] - 0.1)
    
    # Clean up floating point math
    for d in (p_line, p_len, p_shot):
        for k in d:
            d[k] = round(d[k], 2)
            
    return p_line, p_len, p_shot
