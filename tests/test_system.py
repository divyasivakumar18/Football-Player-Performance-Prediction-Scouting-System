"""
Comprehensive Pytest Test Suite for Football IQ Platform.
Tests data cleaning, feature engineering, ML inference, potential scoring,
scouting recommendation, strengths/weaknesses detection, and SQLite database.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_processing import clean_player_data
from src.feature_engineering import (
    compute_per_90_metrics,
    calculate_tactical_scores,
    calculate_overall_performance_score,
    calculate_potential_score,
    engineer_all_features
)
from src.prediction import PerformancePredictionEngine
from src.scouting import (
    calculate_scouting_match_score,
    recommend_scouted_players,
    recommend_alternative_position,
    compute_strength_weaknesses,
    find_hidden_gems
)
from src.database import FootballDatabase


@pytest.fixture
def sample_raw_df():
    """Generates a small synthetic DataFrame for test assertions."""
    return pd.DataFrame([
        {
            "player_id": "P101",
            "player_name": "Test Striker",
            "age": 22,
            "nationality": "France",
            "position": "FW",
            "detailed_position": "ST",
            "club": "Real Madrid",
            "league": "La Liga",
            "season": "2024-2025",
            "appearances": 30,
            "minutes_played": 2500,
            "starts": 28,
            "goals": 20,
            "assists": 8,
            "shots": 80,
            "shots_on_target": 45,
            "passes": 700,
            "pass_accuracy": 82.0,
            "key_passes": 35,
            "chances_created": 40,
            "tackles": 15,
            "interceptions": 10,
            "clearances": 8,
            "blocks": 4,
            "recoveries": 60,
            "aerial_duels_won": 25,
            "defensive_actions": 37,
            "dribbles": 60,
            "successful_dribbles": 40,
            "progressive_carries": 70,
            "progressive_passes": 40,
            "ball_losses": 50,
            "distance_covered": 260.0,
            "sprint_count": 600,
            "fouls": 20,
            "yellow_cards": 2,
            "red_cards": 0,
            "saves": 0,
            "save_percentage": 0.0,
            "clean_sheets": 0,
            "goals_conceded": 0,
            "penalty_saves": 0,
            "player_rating": 86.5,
            "market_value": 75.0,
            "injury_days": 5,
            "previous_season_rating": 84.0,
            "future_performance_rating": 88.0
        },
        {
            "player_id": "P102",
            "player_name": "Test Midfielder",
            "age": 20,
            "nationality": "England",
            "position": "MF",
            "detailed_position": "CM",
            "club": "Arsenal",
            "league": "Premier League",
            "season": "2024-2025",
            "appearances": 25,
            "minutes_played": 2100,
            "starts": 23,
            "goals": 5,
            "assists": 9,
            "shots": 30,
            "shots_on_target": 12,
            "passes": 1600,
            "pass_accuracy": 89.5,
            "key_passes": 55,
            "chances_created": 65,
            "tackles": 50,
            "interceptions": 40,
            "clearances": 20,
            "blocks": 10,
            "recoveries": 150,
            "aerial_duels_won": 20,
            "defensive_actions": 120,
            "dribbles": 40,
            "successful_dribbles": 28,
            "progressive_carries": 65,
            "progressive_passes": 130,
            "ball_losses": 35,
            "distance_covered": 240.0,
            "sprint_count": 480,
            "fouls": 18,
            "yellow_cards": 3,
            "red_cards": 0,
            "saves": 0,
            "save_percentage": 0.0,
            "clean_sheets": 0,
            "goals_conceded": 0,
            "penalty_saves": 0,
            "player_rating": 82.0,
            "market_value": 30.0,
            "injury_days": 0,
            "previous_season_rating": 78.5,
            "future_performance_rating": 85.5
        }
    ])


# 1. Test Data Cleaning
def test_data_cleaning(sample_raw_df):
    clean_df, report = clean_player_data(sample_raw_df)
    assert len(clean_df) == 2
    assert report["status"] == "PASSED_ALL_QUALITY_CHECKS"
    assert clean_df["goals"].min() >= 0
    assert clean_df["pass_accuracy"].max() <= 100.0
    assert clean_df["position"].isin(["FW", "MF", "DF", "GK"]).all()


# 2. Test Feature Engineering
def test_feature_engineering(sample_raw_df):
    clean_df, _ = clean_player_data(sample_raw_df)
    eng_df = engineer_all_features(clean_df)
    
    assert "goals_per_90" in eng_df.columns
    assert "attacking_score" in eng_df.columns
    assert "defensive_score" in eng_df.columns
    assert "passing_score" in eng_df.columns
    assert "overall_performance_score" in eng_df.columns
    assert "potential_score" in eng_df.columns
    assert "potential_tier" in eng_df.columns
    
    # Check bounds
    assert (eng_df["attacking_score"] >= 0).all() and (eng_df["attacking_score"] <= 100).all()
    assert (eng_df["defensive_score"] >= 0).all() and (eng_df["defensive_score"] <= 100).all()
    assert (eng_df["potential_score"] >= 0).all() and (eng_df["potential_score"] <= 100).all()


# 3. Test Machine Learning Inference
def test_prediction_engine():
    engine = PerformancePredictionEngine()
    
    sample_input = {
        "player_name": "Test Star",
        "age": 21,
        "position": "FW",
        "player_rating": 80.0,
        "previous_season_rating": 78.0,
        "minutes_played": 2200,
        "appearances": 28,
        "goals": 15,
        "assists": 7,
        "shots_on_target": 30,
        "pass_accuracy": 82.0,
        "key_passes": 35,
        "tackles": 18
    }
    
    pred_res = engine.predict_single(sample_input)
    assert "predicted_rating" in pred_res
    assert "confidence_lower" in pred_res
    assert "confidence_upper" in pred_res
    assert "potential_score" in pred_res
    assert "potential_tier" in pred_res
    assert "top_factors" in pred_res
    
    assert 50.0 <= pred_res["predicted_rating"] <= 99.0
    assert pred_res["confidence_lower"] <= pred_res["predicted_rating"] <= pred_res["confidence_upper"]


# 4. Test Scouting Recommendations
def test_scouting_recommendations(sample_raw_df):
    clean_df, _ = clean_player_data(sample_raw_df)
    eng_df = engineer_all_features(clean_df)
    
    # Test Match Score
    player_0 = eng_df.iloc[0]
    score = calculate_scouting_match_score(player_0, target_pos="FW", min_rating=80.0, max_budget=100.0)
    assert 0.0 <= score <= 100.0
    
    # Test Shortlist
    recs = recommend_scouted_players(eng_df, target_pos="All", min_rating=75.0, top_n=5)
    assert len(recs) > 0
    assert "scouting_match_score" in recs.columns


# 5. Test Alternative Position & Strengths/Weaknesses
def test_position_and_strength_analysis(sample_raw_df):
    clean_df, _ = clean_player_data(sample_raw_df)
    eng_df = engineer_all_features(clean_df)
    
    p_row = eng_df.iloc[0]
    pos_recs = recommend_alternative_position(p_row)
    assert "primary_position" in pos_recs
    assert "recommended_alternative" in pos_recs
    assert "tactical_reason" in pos_recs
    
    sw = compute_strength_weaknesses(p_row, eng_df)
    assert "strengths" in sw
    assert "weaknesses" in sw
    assert len(sw["strengths"]) > 0


# 6. Test Hidden Gem Detection
def test_hidden_gem_detection(sample_raw_df):
    clean_df, _ = clean_player_data(sample_raw_df)
    eng_df = engineer_all_features(clean_df)
    
    gems = find_hidden_gems(eng_df, max_age=23, max_mv=50.0, min_pot=75.0)
    assert isinstance(gems, pd.DataFrame)
    if not gems.empty:
        assert (gems["age"] <= 23).all()
        assert (gems["market_value"] <= 50.0).all()


# 7. Test SQLite Database
def test_sqlite_database(tmp_path, sample_raw_df):
    clean_df, _ = clean_player_data(sample_raw_df)
    eng_df = engineer_all_features(clean_df)
    
    test_db_file = os.path.join(tmp_path, "test_scouting.db")
    test_db = FootballDatabase(db_path=test_db_file)
    
    test_db.populate_from_dataframe(eng_df)
    
    # Query test
    top_prospects = test_db.get_top_young_prospects(season="2024-2025", max_age=23, limit=5)
    assert len(top_prospects) > 0
    assert "player_name" in top_prospects.columns
    
    # Raw SQL query test
    res = test_db.run_query("SELECT COUNT(*) as cnt FROM players")
    assert res["cnt"].iloc[0] == 2
