import os
import json
import pandas as pd
from src.db import get_connection, init_db

def parse_cricsheet_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    info = data.get('info', {})
    match_id = os.path.basename(file_path).split('.')[0]
    
    # Extract match info
    teams = info.get('teams', ['Unknown', 'Unknown'])
    date = info.get('dates', [''])[0] if info.get('dates') else ''
    venue = info.get('venue', '')
    outcome = info.get('outcome', {})
    winner = outcome.get('winner', 'Tie/No Result')
    
    match_record = {
        'match_id': match_id,
        'date': date,
        'venue': venue,
        'team1': teams[0],
        'team2': teams[1] if len(teams) > 1 else 'Unknown',
        'winner': winner
    }
    
    deliveries_list = []
    innings = data.get('innings', [])
    for idx, inning_data in enumerate(innings):
        # Cricsheet format sometimes has a dict with 'team' and 'overs'
        inning_num = idx + 1
        overs = inning_data.get('overs', [])
        for over in overs:
            over_num = over.get('over', 0)
            for ball_idx, ball in enumerate(over.get('deliveries', [])):
                runs = ball.get('runs', {})
                wicket = ball.get('wickets', [{}])[0] if 'wickets' in ball else {}
                
                delivery_record = {
                    'match_id': match_id,
                    'inning': inning_num,
                    'over_num': over_num,
                    'ball_num': ball_idx + 1,
                    'batter': ball.get('batter', ''),
                    'bowler': ball.get('bowler', ''),
                    'non_striker': ball.get('non_striker', ''),
                    'runs_batter': runs.get('batter', 0),
                    'runs_extras': runs.get('extras', 0),
                    'runs_total': runs.get('total', 0),
                    'wicket_type': wicket.get('kind', ''),
                    'player_out': wicket.get('player_out', '')
                }
                deliveries_list.append(delivery_record)
                
    return match_record, deliveries_list

def ingest_all_jsons(raw_data_dir):
    init_db()
    conn = get_connection()
    
    matches_data = []
    deliveries_data = []
    
    for filename in os.listdir(raw_data_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(raw_data_dir, filename)
            try:
                match_rec, deliv_list = parse_cricsheet_json(file_path)
                matches_data.append(match_rec)
                deliveries_data.extend(deliv_list)
            except Exception as e:
                print(f"Error parsing {filename}: {e}")
                
    if matches_data:
        matches_df = pd.DataFrame(matches_data)
        # Drop duplicates if ingesting multiple times for safety
        matches_df.drop_duplicates(subset=['match_id'], inplace=True)
        matches_df.to_sql('matches', conn, if_exists='append', index=False)
        print(f"Inserted {len(matches_df)} matches.")
        
    if deliveries_data:
        deliveries_df = pd.DataFrame(deliveries_data)
        deliveries_df.to_sql('deliveries', conn, if_exists='append', index=False)
        print(f"Inserted {len(deliveries_df)} deliveries.")
        
    conn.close()

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
    print(f"Scanning for JSONs in {data_dir}...")
    ingest_all_jsons(data_dir)
