import sqlite3
import os
import sys

# Import database path from config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import DB_PATH

def get_db_connection(db_path=DB_PATH):
    """Establishes and returns a connection to the SQLite database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    # Enable foreign keys support in SQLite
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db(db_path=DB_PATH):
    """Creates the database schema tables if they do not already exist."""
    print(f"Initializing database at: {db_path}")
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # 1. Matches Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS matches (
        match_id TEXT PRIMARY KEY,
        date TEXT NOT NULL,
        team1 TEXT NOT NULL,
        team2 TEXT NOT NULL,
        venue TEXT NOT NULL,
        toss_winner TEXT,
        toss_decision TEXT,
        winner TEXT,
        margin_type TEXT,
        margin_value INTEGER
    );
    """)

    # 2. Players Table (lookup for styles)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS players (
        name TEXT PRIMARY KEY,
        batting_hand TEXT NOT NULL,         -- 'Right-hand bat', 'Left-hand bat'
        bowling_hand TEXT NOT NULL,         -- 'Right-arm', 'Left-arm'
        bowling_sub_style TEXT NOT NULL     -- 'pace', 'off-spin', 'leg-spin', 'chinaman'
    );
    """)

    # 3. Deliveries Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS deliveries (
        delivery_id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_id TEXT NOT NULL,
        innings INTEGER NOT NULL,            -- 1 or 2
        over INTEGER NOT NULL,               -- 0 to 19
        ball INTEGER NOT NULL,               -- 1 to 6 (or more for extras)
        batsman TEXT NOT NULL,
        bowler TEXT NOT NULL,
        non_striker TEXT NOT NULL,
        runs_batsman INTEGER NOT NULL,       -- Runs scored by batsman off bat
        runs_extras INTEGER NOT NULL,        -- Extra runs (wides, no-balls, etc.)
        runs_total INTEGER NOT NULL,         -- Total runs scored on the delivery
        extra_type TEXT,                     -- 'wides', 'noballs', 'legbyes', 'byes', NULL
        dismissal_player TEXT,               -- Name of player dismissed, NULL if no wicket
        dismissal_kind TEXT,                 -- 'caught', 'bowled', 'lbw', 'run out', etc., NULL
        match_phase TEXT NOT NULL,           -- 'powerplay', 'middle', 'death'
        line TEXT DEFAULT 'unknown',         -- Extracted from NLP commentary
        length TEXT DEFAULT 'unknown',       -- Extracted from NLP commentary
        shot_intent TEXT DEFAULT 'unknown',  -- Extracted from NLP commentary
        FOREIGN KEY (match_id) REFERENCES matches(match_id),
        FOREIGN KEY (batsman) REFERENCES players(name),
        FOREIGN KEY (bowler) REFERENCES players(name)
    );
    """)

    # Create Indexes for sequential query speed
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_deliveries_match ON deliveries(match_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_deliveries_batsman ON deliveries(batsman);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_deliveries_bowler ON deliveries(bowler);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_deliveries_seq ON deliveries(match_id, innings, over, ball);")

    conn.commit()
    conn.close()
    print("Database tables initialized successfully with indexes.")

if __name__ == "__main__":
    init_db()
