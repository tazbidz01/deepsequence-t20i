# DeepSequence-T20I Database Schema

This document outlines the Entity-Relationship (ER) model for the SQLite database powering the DeepSequence-T20I backend (`t20i_engine.db`).

```mermaid
erDiagram
    MATCHES {
        TEXT match_id PK "Primary Key (e.g., '1001349')"
        TEXT date "Match Date"
        TEXT venue "Stadium/Location"
        TEXT team1 "First Team Name"
        TEXT team2 "Second Team Name"
        TEXT winner "Winning Team Name"
    }

    DELIVERIES {
        INTEGER id PK "Auto-incrementing ID"
        TEXT match_id FK "Foreign Key -> MATCHES(match_id)"
        INTEGER inning "Inning Number (1 or 2)"
        INTEGER over_num "Over Number (0-19)"
        INTEGER ball_num "Ball Number in Over"
        TEXT batter "Batsman on strike"
        TEXT bowler "Bowler"
        TEXT non_striker "Batsman off strike"
        INTEGER runs_batter "Runs scored off the bat"
        INTEGER runs_extras "Extra runs (wides, no-balls)"
        INTEGER runs_total "Total runs for the delivery"
        TEXT wicket_type "How the wicket fell (if any)"
        TEXT player_out "Name of the player dismissed"
    }

    PLAYERS {
        TEXT cricsheet_id PK "Primary Key (Cricsheet Identifier)"
        TEXT name "Player's Name"
        TEXT cricinfo_id "ESPNcricinfo mapped ID"
        TEXT cricbuzz_id "Cricbuzz mapped ID"
    }

    %% Relationships
    MATCHES ||--o{ DELIVERIES : "contains (1 match has many deliveries)"
    PLAYERS |o--o{ DELIVERIES : "bats/bowls in (implicit link via name)"
```

## Tables Breakdown
- **MATCHES**: The root table tracking high-level match metadata.
- **DELIVERIES**: The core sequence table that stores ball-by-ball events. This is queried by the PyTorch LSTM to formulate ML input tensors.
- **PLAYERS**: The global dictionary registry populated from Cricsheet's `people.csv`. It cross-references with the `DELIVERIES` table to fetch live ESPNcricinfo profile URLs for the Streamlit dashboard.
