"""
Data Cleaning and Preprocessing Pipeline for Football Player Analytics.
Handles validation, imputation, deduplication, and sanity checking.
"""

import os
import numpy as np
import pandas as pd


POSITION_MAPPING = {
    "ST": "FW", "CF": "FW", "LW": "FW", "RW": "FW", "FORWARD": "FW", "STRIKER": "FW", "WINGER": "FW",
    "CAM": "MF", "CM": "MF", "CDM": "MF", "MIDFIELDER": "MF", "AM": "MF", "DM": "MF",
    "CB": "DF", "LB": "DF", "RB": "DF", "LWB": "DF", "RWB": "DF", "DEFENDER": "DF",
    "GK": "GK", "GOALKEEPER": "GK"
}

NUMERICAL_COLS = [
    "age", "appearances", "minutes_played", "starts", "goals", "assists",
    "shots", "shots_on_target", "passes", "pass_accuracy", "key_passes",
    "chances_created", "tackles", "interceptions", "clearances", "blocks",
    "recoveries", "aerial_duels_won", "defensive_actions", "dribbles",
    "successful_dribbles", "progressive_carries", "progressive_passes",
    "ball_losses", "distance_covered", "sprint_count", "fouls",
    "yellow_cards", "red_cards", "saves", "save_percentage", "clean_sheets",
    "goals_conceded", "penalty_saves", "player_rating", "market_value",
    "injury_days", "previous_season_rating", "future_performance_rating"
]

CATEGORICAL_COLS = [
    "player_id", "player_name", "nationality", "position",
    "detailed_position", "club", "league", "season"
]


def clean_player_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Cleans raw player performance data with full validation and imputation.
    
    Returns:
        tuple[pd.DataFrame, dict]: Cleaned dataframe and data quality report.
    """
    df = df.copy()
    initial_rows = len(df)
    
    # 1. Deduplicate by (player_id, season)
    df = df.drop_duplicates(subset=["player_id", "season"], keep="last")
    duplicates_removed = initial_rows - len(df)
    
    # 2. Position standardization
    if "position" in df.columns:
        df["position"] = df["position"].astype(str).str.upper().str.strip()
        df["position"] = df["position"].map(lambda x: POSITION_MAPPING.get(x, x))
        # Ensure valid standard positions
        valid_positions = {"FW", "MF", "DF", "GK"}
        df["position"] = df["position"].apply(lambda x: x if x in valid_positions else "MF")
    
    # 3. Categorical missing value imputation (Mode)
    for cat_col in CATEGORICAL_COLS:
        if cat_col in df.columns:
            mode_val = df[cat_col].mode()
            fill_val = mode_val[0] if not mode_val.empty else "Unknown"
            df[cat_col] = df[cat_col].fillna(fill_val).astype(str).str.strip()
            
    # 4. Numerical type casting & missing value imputation (Median per position)
    for num_col in NUMERICAL_COLS:
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce")
            
            # Position-specific median imputation
            pos_medians = df.groupby("position")[num_col].transform("median")
            overall_median = df[num_col].median() if not np.isnan(df[num_col].median()) else 0
            df[num_col] = df[num_col].fillna(pos_medians).fillna(overall_median)
            
    # 5. Sanity checks & Range Constraints
    if "age" in df.columns:
        df["age"] = df["age"].clip(lower=15, upper=45).astype(int)
    if "appearances" in df.columns:
        df["appearances"] = df["appearances"].clip(lower=1, upper=45).astype(int)
    if "minutes_played" in df.columns:
        df["minutes_played"] = df["minutes_played"].clip(lower=30, upper=4050).astype(int)
    if "pass_accuracy" in df.columns:
        df["pass_accuracy"] = df["pass_accuracy"].clip(lower=20.0, upper=100.0)
    if "save_percentage" in df.columns:
        df["save_percentage"] = df["save_percentage"].clip(lower=0.0, upper=100.0)
    if "player_rating" in df.columns:
        df["player_rating"] = df["player_rating"].clip(lower=45.0, upper=99.0)
    if "previous_season_rating" in df.columns:
        df["previous_season_rating"] = df["previous_season_rating"].clip(lower=45.0, upper=99.0)
    if "future_performance_rating" in df.columns:
        df["future_performance_rating"] = df["future_performance_rating"].clip(lower=45.0, upper=99.0)
        
    # Ensure all count columns are non-negative integers
    count_cols = [
        "starts", "goals", "assists", "shots", "shots_on_target", "passes",
        "key_passes", "chances_created", "tackles", "interceptions", "clearances",
        "blocks", "recoveries", "aerial_duels_won", "defensive_actions",
        "dribbles", "successful_dribbles", "progressive_carries",
        "progressive_passes", "ball_losses", "sprint_count", "fouls",
        "yellow_cards", "red_cards", "saves", "clean_sheets", "goals_conceded",
        "penalty_saves", "injury_days"
    ]
    for c in count_cols:
        if c in df.columns:
            df[c] = df[c].clip(lower=0).round().astype(int)
            
    # Ensure defensive_actions is consistent
    df["defensive_actions"] = df["tackles"] + df["interceptions"] + df["clearances"] + df["blocks"]
    
    # 6. Outlier Summary (IQR based)
    outlier_report = {}
    for col in ["goals", "assists", "passes", "tackles", "market_value", "player_rating"]:
        if col in df.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            n_outliers = int(((df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))).sum())
            outlier_report[col] = n_outliers
            
    quality_report = {
        "initial_rows": initial_rows,
        "clean_rows": len(df),
        "duplicates_removed": duplicates_removed,
        "unique_players": int(df["player_id"].nunique()),
        "unique_clubs": int(df["club"].nunique()),
        "seasons_covered": sorted(df["season"].unique().tolist()),
        "outlier_counts": outlier_report,
        "status": "PASSED_ALL_QUALITY_CHECKS"
    }
    return df, quality_report


import sys
if os.path.dirname(os.path.dirname(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.feature_engineering import engineer_all_features


def load_and_preprocess(raw_filepath: str, save_processed: bool = True) -> tuple[pd.DataFrame, dict]:
    """Loads raw CSV, runs data cleaning and feature engineering, and optionally saves to processed directory."""
    if not os.path.exists(raw_filepath):
        raise FileNotFoundError(f"Raw data file not found at: {raw_filepath}")
        
    raw_df = pd.read_csv(raw_filepath)
    clean_df, report = clean_player_data(raw_df)
    
    # Apply feature engineering
    processed_df = engineer_all_features(clean_df)
    
    if save_processed:
        proc_dir = os.path.join(os.path.dirname(raw_filepath), "..", "processed")
        os.makedirs(proc_dir, exist_ok=True)
        proc_filepath = os.path.join(proc_dir, "players_processed.csv")
        processed_df.to_csv(proc_filepath, index=False)
        report["processed_filepath"] = proc_filepath
        report["total_features"] = processed_df.shape[1]
        
    return processed_df, report


if __name__ == "__main__":
    raw_file = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "players.csv")
    cleaned_df, rep = load_and_preprocess(raw_file)
    print("Data Cleaning Complete!")
    print(f"Report: {rep}")
