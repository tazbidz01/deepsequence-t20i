"""
DeepSequence-T20I Configuration File
Defines sequence sizes, padding constants, categorical encoding dimensions,
hyperparameters, and directory paths.
"""

import os

# --- Core Directories ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
DB_PATH = os.path.join(PROCESSED_DATA_DIR, "t20i_engine.db")

# Ensure directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# --- Model Sequence Constraints ---
SEQUENCE_LENGTH = 6  # Rolling window size (number of preceding balls faced)
PADDING_VALUE = 0.0

# --- Categorical Variables Mappings ---
# Mappings map category names to integer IDs for embedding layers.
MATCH_PHASES = {
    "powerplay": 0,  # Overs 0-6
    "middle": 1,     # Overs 7-15
    "death": 2       # Overs 16-20
}

BOWLER_HANDS = {
    "right-arm": 0,
    "left-arm": 1
}

BOWLER_STYLES = {
    "pace": 0,       # Fast, Fast-Medium, Medium-Fast, Medium
    "off-spin": 1,   # Off-break
    "leg-spin": 2,   # Leg-break
    "chinaman": 3    # Left-arm unorthodox
}

BALL_LINES = {
    "outside_off": 0,
    "middle_off": 1,
    "middle": 2,
    "middle_leg": 3,
    "down_leg": 4,
    "unknown": 5
}

BALL_LENGTHS = {
    "yorker": 0,
    "full": 1,
    "slot": 2,
    "good_length": 3,
    "short": 4,
    "full_toss": 5,
    "unknown": 6
}

SHOT_INTENTS = {
    "drive": 0,
    "pull": 1,
    "cut": 2,
    "sweep": 3,
    "flick": 4,
    "block": 5,
    "loft": 6,
    "leave": 7,
    "unknown": 8
}

# --- Embedding Dimensions Configuration ---
# Format: (num_classes, embedding_dim)
EMBEDDING_CONFIGS = {
    "match_phase": (len(MATCH_PHASES), 2),
    "bowler_hand": (len(BOWLER_HANDS), 2),
    "bowler_style": (len(BOWLER_STYLES), 2),
    "line": (len(BALL_LINES), 3),
    "length": (len(BALL_LENGTHS), 3),
    "shot_intent": (len(SHOT_INTENTS), 4),
    "prev_wicket": (2, 2)
}

TOTAL_EMBEDDING_DIM = sum(dim for _, dim in EMBEDDING_CONFIGS.values()) # 2+2+2+3+3+4+2 = 18

# --- Numerical (Continuous) Feature Dimensions ---
NUMERICAL_FEATURE_DIM = 7  # innings_ball, batsman_score, batsman_balls, last_6_runs, dot_streak, speed_class, innings_rr

# --- Total Model Input Dimension ---
MODEL_INPUT_DIM = TOTAL_EMBEDDING_DIM + NUMERICAL_FEATURE_DIM  # 18 + 7 = 25

# --- Hyperparameters ---
MODEL_HYPERPARAMS = {
    "hidden_dim": 64,
    "lstm_layers": 2,
    "dropout": 0.2,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 20,
    "focal_alpha": 0.92, # Balance coefficient to prioritize dismissal events
    "focal_gamma": 2.0   # Focus parameter to downweight easy-to-classify runs
}
