import os
import sqlite3
import pandas as pd
from datasets import load_dataset
from src.nlp import CommentaryParser

def get_top_batsmen(db_path, limit=30):
    """Fetch the top N run-scorers from our T20I database."""
    conn = sqlite3.connect(db_path)
    query = f"""
    SELECT batter, SUM(runs_batter) as total_runs, COUNT(runs_batter) as balls_faced
    FROM deliveries
    GROUP BY batter
    HAVING balls_faced > 50
    ORDER BY total_runs DESC
    LIMIT {limit}
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df['batter'].tolist()

def analyze_all_players():
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "t20i_engine.db")
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return
        
    top_batsmen = get_top_batsmen(db_path, limit=30)
    print(f"Loaded top {len(top_batsmen)} batsmen from local DB.")
    
    print("Loading HuggingFace dataset VinitT/Cricket-Commentary-Sample...")
    dataset = load_dataset("VinitT/Cricket-Commentary-Sample", split="train")
    df_hf = pd.DataFrame(dataset)
    
    parser = CommentaryParser()
    
    results = []
    
    for player in top_batsmen:
        # Some player names in DB might be "V Kohli", we might need just "Kohli" or the full name
        # The HF dataset usually uses names like "Virat Kohli" or "V Kohli". We will use the last name 
        # to ensure we get matches, but that might clash with common names (like "Smith").
        # To be safe, we'll try the exact DB string first, then just the last name if we get 0.
        last_name = player.split()[-1]
        
        # Filter HF dataset for this player
        player_df = df_hf[df_hf['input'].str.contains(last_name, case=False, na=False)].copy()
        
        if len(player_df) == 0:
            continue
            
        # Isolate dismissals
        dismissals = player_df[player_df['input'].str.contains('dismissal is True', case=False, na=False)]
        
        if len(dismissals) == 0:
            continue
            
        lines = []
        lengths = []
        
        for idx, row in dismissals.iterrows():
            commentary = str(row['output']).strip()
            features = parser.extract_features(commentary)
            
            line = features['line'].split(" (")[0]
            length = features['length'].split(" (")[0]
            
            if line != "Unknown": lines.append(line)
            if length != "Unknown": lengths.append(length)
            
        if not lines or not lengths:
            continue
            
        # Calculate Primary Weaknesses (Mode)
        primary_line = pd.Series(lines).mode().iloc[0] if len(pd.Series(lines).mode()) > 0 else "Unknown"
        primary_length = pd.Series(lengths).mode().iloc[0] if len(pd.Series(lengths).mode()) > 0 else "Unknown"
        
        results.append({
            'Player': player,
            'HF_Deliveries_Found': len(player_df),
            'HF_Dismissals_Found': len(dismissals),
            'Primary_Weakness_Line': primary_line,
            'Primary_Weakness_Length': primary_length
        })
        
    df_results = pd.DataFrame(results)
    
    # Save the generalized dataset
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "global_vulnerabilities.csv")
    df_results.to_csv(out_path, index=False)
    
    print("\n--- GLOBAL PLAYER VULNERABILITIES ---")
    print(df_results.to_string(index=False))
    print(f"\nSaved massive vulnerability matrix to {out_path}")

if __name__ == "__main__":
    analyze_all_players()
