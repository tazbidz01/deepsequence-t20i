import os
import pandas as pd
from src.db import get_connection

def fetch_and_store_registry():
    url = "https://cricsheet.org/register/people.csv"
    print(f"Downloading player registry from {url}...")
    
    try:
        # Load CSV into pandas
        df = pd.read_csv(url)
        
        if 'identifier' in df.columns:
            # Select and rename required columns to match DB schema
            cols_to_extract = {'identifier': 'cricsheet_id', 'name': 'name'}
            if 'key_cricinfo' in df.columns:
                cols_to_extract['key_cricinfo'] = 'cricinfo_id'
            if 'key_cricbuzz' in df.columns:
                cols_to_extract['key_cricbuzz'] = 'cricbuzz_id'
                
            players_df = df[list(cols_to_extract.keys())].rename(columns=cols_to_extract)
            
            # Drop rows without a valid Cricsheet ID
            players_df.dropna(subset=['cricsheet_id'], inplace=True)
            
            conn = get_connection()
            
            # Refresh master table
            cursor = conn.cursor()
            cursor.execute("DELETE FROM players")
            conn.commit()
            
            players_df.to_sql('players', conn, if_exists='append', index=False)
            conn.close()
            
            print(f"Successfully inserted {len(players_df)} players into the database.")
        else:
            print("Error: The CSV does not contain the 'identifier' column.")
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    fetch_and_store_registry()
