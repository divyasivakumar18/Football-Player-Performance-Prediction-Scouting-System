"""
Feature Engineering Module for Football Player Performance and Scouting.
Calculates domain-specific tactical metrics, composite scores, potential ratings, and archetypes.
"""

import numpy as np
import pandas as pd


def _min_max_scale_series(s: pd.Series, low: float = 0.0, high: float = 100.0) -> pd.Series:
    """Scales a pandas series to [low, high] safely."""
    s_min = s.min()
    s_max = s.max()
    if s_max == s_min:
        return pd.Series(50.0, index=s.index)
    scaled = (s - s_min) / (s_max - s_min) * (high - low) + low
    return scaled.clip(low, high)


def compute_per_90_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates per-90 normalized statistics."""
    df = df.copy()
    nineties = np.maximum(df["minutes_played"] / 90.0, 1.0)
    
    df["goals_per_90"] = (df["goals"] / nineties).round(2)
    df["assists_per_90"] = (df["assists"] / nineties).round(2)
    df["goal_contributions_per_90"] = ((df["goals"] + df["assists"]) / nineties).round(2)
    df["shots_per_90"] = (df["shots"] / nineties).round(2)
    df["shots_on_target_per_90"] = (df["shots_on_target"] / nineties).round(2)
    df["key_passes_per_90"] = (df["key_passes"] / nineties).round(2)
    df["chances_created_per_90"] = (df["chances_created"] / nineties).round(2)
    df["passes_per_90"] = (df["passes"] / nineties).round(1)
    
    df["tackles_per_90"] = (df["tackles"] / nineties).round(2)
    df["interceptions_per_90"] = (df["interceptions"] / nineties).round(2)
    df["clearances_per_90"] = (df["clearances"] / nineties).round(2)
    df["recoveries_per_90"] = (df["recoveries"] / nineties).round(2)
    df["defensive_actions_per_90"] = (df["defensive_actions"] / nineties).round(2)
    
    df["dribbles_per_90"] = (df["dribbles"] / nineties).round(2)
    df["successful_dribbles_per_90"] = (df["successful_dribbles"] / nineties).round(2)
    df["progressive_carries_per_90"] = (df["progressive_carries"] / nineties).round(2)
    df["progressive_passes_per_90"] = (df["progressive_passes"] / nineties).round(2)
    
    df["shot_accuracy"] = np.where(df["shots"] > 0, (df["shots_on_target"] / df["shots"] * 100).round(1), 0.0)
    df["dribble_success_rate"] = np.where(df["dribbles"] > 0, (df["successful_dribbles"] / df["dribbles"] * 100).round(1), 0.0)
    
    return df


def calculate_tactical_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Computes attacking, defensive, passing, and progression composite scores."""
    df = df.copy()
    
    # 1. Attacking Score (0-100)
    raw_atk = (
        df["goals_per_90"] * 35.0 +
        df["assists_per_90"] * 25.0 +
        df["shots_on_target_per_90"] * 15.0 +
        df["key_passes_per_90"] * 15.0 +
        df["chances_created_per_90"] * 10.0
    )
    df["attacking_score"] = _min_max_scale_series(raw_atk, 15.0, 99.0).round(1)
    
    # 2. Defensive Score (0-100)
    raw_def = (
        df["tackles_per_90"] * 25.0 +
        df["interceptions_per_90"] * 25.0 +
        df["clearances_per_90"] * 20.0 +
        df["recoveries_per_90"] * 15.0 +
        (df["aerial_duels_won"] / np.maximum(df["minutes_played"] / 90.0, 1.0)) * 15.0
    )
    df["defensive_score"] = _min_max_scale_series(raw_def, 15.0, 99.0).round(1)
    
    # 3. Passing Score (0-100)
    raw_pass = (
        (df["pass_accuracy"] - 50.0).clip(lower=0) * 0.40 +
        df["key_passes_per_90"] * 25.0 +
        df["progressive_passes_per_90"] * 20.0 +
        df["chances_created_per_90"] * 15.0
    )
    df["passing_score"] = _min_max_scale_series(raw_pass, 20.0, 99.0).round(1)
    
    # 4. Progression Score (0-100)
    raw_prog = (
        df["successful_dribbles_per_90"] * 35.0 +
        df["progressive_carries_per_90"] * 35.0 +
        df["progressive_passes_per_90"] * 30.0
    )
    df["progression_score"] = _min_max_scale_series(raw_prog, 15.0, 99.0).round(1)
    
    # 5. Goalkeeping Score (0-100)
    raw_gk = (
        df["save_percentage"] * 0.40 +
        (df["clean_sheets"] / np.maximum(df["appearances"], 1) * 100) * 0.30 +
        (df["saves"] / np.maximum(df["appearances"], 1) * 10) * 0.20 +
        df["penalty_saves"] * 2.0
    )
    df["goalkeeping_score"] = _min_max_scale_series(raw_gk, 20.0, 99.0).round(1)
    
    return df


def calculate_overall_performance_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a position-weighted overall performance score from 0-100.
    """
    df = df.copy()
    
    def compute_pos_score(row):
        pos = row["position"]
        if pos == "FW":
            score = (
                0.45 * row["attacking_score"] +
                0.25 * row["progression_score"] +
                0.20 * row["passing_score"] +
                0.10 * row["defensive_score"]
            )
        elif pos == "MF":
            score = (
                0.35 * row["passing_score"] +
                0.25 * row["progression_score"] +
                0.20 * row["attacking_score"] +
                0.20 * row["defensive_score"]
            )
        elif pos == "DF":
            score = (
                0.50 * row["defensive_score"] +
                0.25 * row["passing_score"] +
                0.15 * row["progression_score"] +
                0.10 * row["attacking_score"]
            )
        elif pos == "GK":
            score = (
                0.80 * row["goalkeeping_score"] +
                0.20 * row["passing_score"]
            )
        else:
            score = (row["attacking_score"] + row["defensive_score"] + row["passing_score"] + row["progression_score"]) / 4.0
            
        return round(float(score), 1)

    df["overall_performance_score"] = df.apply(compute_pos_score, axis=1)
    return df


def calculate_potential_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a realistic potential score (0-100) and classifies players into prospect tiers.
    """
    df = df.copy()
    
    target_rating = df["future_performance_rating"] if "future_performance_rating" in df.columns else df["player_rating"]
    current_rating = df["player_rating"]
    age = df["age"]
    minutes = df["minutes_played"]
    
    # Age factor: younger players get higher growth multipliers
    age_multiplier = np.where(
        age <= 20, 1.15,
        np.where(age <= 23, 1.08,
        np.where(age <= 26, 1.02,
        np.where(age <= 29, 0.96, 0.88)))
    )
    
    # Playing time experience factor
    exp_factor = np.clip(minutes / 2000.0, 0.75, 1.05)
    
    # Potential formula
    raw_pot = (
        0.40 * target_rating +
        0.25 * current_rating +
        0.20 * (target_rating * age_multiplier) +
        0.15 * (current_rating * exp_factor)
    )
    
    df["potential_score"] = np.clip(raw_pot, 45.0, 99.0).round(1)
    
    # Tier classification
    def get_tier(score):
        if score >= 88.0:
            return "Elite Prospect"
        elif score >= 80.0:
            return "High Potential"
        elif score >= 72.0:
            return "Promising"
        elif score >= 64.0:
            return "Development Player"
        else:
            return "Low Priority"
            
    df["potential_tier"] = df["potential_score"].apply(get_tier)
    
    # Hidden Gem flag: young, high potential, high predicted rating, low market value
    df["is_hidden_gem"] = (
        (df["age"] <= 23) &
        (df["potential_score"] >= 78.0) &
        (df["future_performance_rating"] >= 76.0) &
        (df["market_value"] <= 35.0)
    )
    
    return df


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Runs all feature engineering steps sequentially."""
    df = compute_per_90_metrics(df)
    df = calculate_tactical_scores(df)
    df = calculate_overall_performance_score(df)
    df = calculate_potential_score(df)
    return df
