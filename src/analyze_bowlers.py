import os
import sqlite3
import pandas as pd
from datasets import load_dataset
from src.nlp import CommentaryParser

def get_top_bowlers(db_path, limit=30):
    """Fetch the top wicket-takers from our T20I database."""
    conn = sqlite3.connect(db_path)
    query = f"""
    SELECT bowler, COUNT(CASE WHEN wicket_type != '' AND wicket_type != 'run out' THEN 1 END) as total_wickets
    FROM deliveries
    GROUP BY bowler
    HAVING total_wickets > 0
    ORDER BY total_wickets DESC
    LIMIT {limit}
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df['bowler'].tolist()

def analyze_all_bowlers():
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "t20i_engine.db")
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return
        
    top_bowlers = get_top_bowlers(db_path, limit=30)
    print(f"Loaded top {len(top_bowlers)} bowlers from local DB.")
    
    print("Loading HuggingFace dataset VinitT/Cricket-Commentary-Sample...")
    dataset = load_dataset("VinitT/Cricket-Commentary-Sample", split="train")
    df_hf = pd.DataFrame(dataset)
    
    parser = CommentaryParser()
    results = []
    
    for player in top_bowlers:
        last_name = player.split()[-1]
        
        # Filter HF dataset for this bowler
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
            
        # Calculate Primary Strengths (Mode)
        primary_line = pd.Series(lines).mode().iloc[0] if len(pd.Series(lines).mode()) > 0 else "Unknown"
        primary_length = pd.Series(lengths).mode().iloc[0] if len(pd.Series(lengths).mode()) > 0 else "Unknown"
        
        results.append({
            'Player': player,
            'HF_Deliveries_Found': len(player_df),
            'HF_Wickets_Found': len(dismissals),
            'Primary_Strength_Line': primary_line,
            'Primary_Strength_Length': primary_length
        })
        
    df_results = pd.DataFrame(results)
    
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "global_bowler_strengths.csv")
    df_results.to_csv(out_path, index=False)
    
    print("\n--- GLOBAL BOWLER STRENGTHS ---")
    print(df_results.to_string(index=False))
    print(f"\nSaved massive strength matrix to {out_path}")

if __name__ == "__main__":
    analyze_all_bowlers()
