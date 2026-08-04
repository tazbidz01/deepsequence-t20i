import pandas as pd
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

def get_all_batsmen():
    df = load_data("SELECT DISTINCT batter FROM deliveries ORDER BY batter ASC")
    if not df.empty:
        return df['batter'].tolist()
    return ["No Data Available"]

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

def get_player_cricinfo_link(player_name):
    safe_name = player_name.replace("'", "''")
    query = f"SELECT cricinfo_id FROM players WHERE name = '{safe_name}' LIMIT 1"
    df = load_data(query)
    if not df.empty:
        cricinfo_id = df.iloc[0]['cricinfo_id']
        if pd.notna(cricinfo_id):
            return f"https://www.espncricinfo.com/ci/content/player/{int(float(cricinfo_id))}.html"
    return None

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
        return sr, dismissals, balls
    return 0.0, 0, 0
