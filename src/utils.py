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
