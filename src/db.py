import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "t20i_engine.db")

def get_connection():
    """Returns a connection to the SQLite database."""
    return sqlite3.connect(DB_PATH)

def init_db():
    """Initializes the database schemas for matches, deliveries, and players."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_connection()
    cursor = conn.cursor()
    
    # Matches table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS matches (
            match_id TEXT PRIMARY KEY,
            date TEXT,
            venue TEXT,
            team1 TEXT,
            team2 TEXT,
            winner TEXT
        )
    ''')
    
    # Deliveries table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS deliveries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id TEXT,
            inning INTEGER,
            over_num INTEGER,
            ball_num INTEGER,
            batter TEXT,
            bowler TEXT,
            non_striker TEXT,
            runs_batter INTEGER,
            runs_extras INTEGER,
            runs_total INTEGER,
            wicket_type TEXT,
            player_out TEXT,
            FOREIGN KEY(match_id) REFERENCES matches(match_id)
        )
    ''')
    
    # Players registry table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS players (
            cricsheet_id TEXT PRIMARY KEY,
            name TEXT,
            cricinfo_id TEXT,
            cricbuzz_id TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    print("Initializing database...")
    init_db()
    print(f"Database successfully initialized at {DB_PATH}")
