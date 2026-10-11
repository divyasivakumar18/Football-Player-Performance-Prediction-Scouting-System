"""
Prediction Engine and Explainable AI (XAI) for Football Player Performance.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from src.feature_engineering import (
    compute_per_90_metrics,
    calculate_tactical_scores,
    calculate_overall_performance_score
)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")


class PerformancePredictionEngine:
    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        self.model = None
        self.scaler = None
        self.metadata = None
        self.feature_cols = []
        self.rmse = 1.10
        self.load_artifacts()
        
    def load_artifacts(self):
        """Loads serialized model, scaler, and metadata."""
        model_path = os.path.join(self.models_dir, "performance_model.pkl")
        scaler_path = os.path.join(self.models_dir, "scaler.pkl")
        meta_path = os.path.join(self.models_dir, "model_metadata.json")
        
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                self.metadata = json.load(f)
                self.feature_cols = self.metadata.get("feature_cols", [])
                best_model = self.metadata.get("best_model_name", "Gradient Boosting")
                self.rmse = self.metadata.get("model_comparison", {}).get(best_model, {}).get("RMSE", 1.10)

    def prepare_input_features(self, raw_input: dict) -> pd.DataFrame:
        """Ensures all 75 features and per-90 metrics are computed from user inputs."""
        input_data = raw_input.copy()
        
        # Position and defaults
        pos = input_data.get("position", "MF")
        curr_rating = float(input_data.get("player_rating", input_data.get("previous_season_rating", 75.0)))
        input_data["player_rating"] = curr_rating
        
        # Default fallback values for any unspecified field
        defaults = {
            "player_id": "PRED_001",
            "player_name": "Scouted Prospect",
            "age": 22,
            "nationality": "International",
            "position": pos,
            "detailed_position": "CM" if pos == "MF" else ("ST" if pos == "FW" else ("CB" if pos == "DF" else "GK")),
            "club": "Scouting Target",
            "league": "Scouting Network",
            "season": "2024-2025",
            "appearances": 28,
            "minutes_played": 2200,
            "starts": 25,
            "goals": 6 if pos == "FW" else (3 if pos == "MF" else 1),
            "assists": 5 if pos == "FW" else (6 if pos == "MF" else 2),
            "shots": 40 if pos == "FW" else 20,
            "shots_on_target": 20 if pos == "FW" else 8,
            "passes": 1400 if pos in ["MF", "DF"] else 600,
            "pass_accuracy": 82.0,
            "key_passes": 35 if pos in ["FW", "MF"] else 10,
            "chances_created": 40 if pos in ["FW", "MF"] else 12,
            "tackles": 45 if pos in ["MF", "DF"] else 15,
            "interceptions": 35 if pos in ["MF", "DF"] else 10,
            "clearances": 60 if pos == "DF" else 15,
            "blocks": 15 if pos == "DF" else 5,
            "recoveries": 120,
            "aerial_duels_won": 30,
            "defensive_actions": 155 if pos == "DF" else 65,
            "dribbles": 45 if pos in ["FW", "MF"] else 12,
            "successful_dribbles": 28 if pos in ["FW", "MF"] else 8,
            "progressive_carries": 55 if pos in ["FW", "MF"] else 20,
            "progressive_passes": 85 if pos in ["MF", "DF"] else 30,
            "ball_losses": 45,
            "distance_covered": 230.0,
            "sprint_count": 550,
            "fouls": 24,
            "yellow_cards": 3,
            "red_cards": 0,
            "saves": 0 if pos != "GK" else 85,
            "save_percentage": 0.0 if pos != "GK" else 72.0,
            "clean_sheets": 0 if pos != "GK" else 9,
            "goals_conceded": 0 if pos != "GK" else 32,
            "penalty_saves": 0 if pos != "GK" else 1,
            "market_value": 25.0,
            "injury_days": 10,
            "previous_season_rating": curr_rating - 1.0,
            "future_performance_rating": curr_rating
        }
        
        for k, v in defaults.items():
            if k not in input_data:
                input_data[k] = v
                
        df_single = pd.DataFrame([input_data])
        
        # Calculate engineered metrics
        df_single = compute_per_90_metrics(df_single)
        df_single = calculate_tactical_scores(df_single)
        df_single = calculate_overall_performance_score(df_single)
        
        return df_single

    def predict_single(self, input_dict: dict) -> dict:
        """
        Runs ML prediction, confidence interval calculation, potential score, and explainability factors.
        """
        if self.model is None:
            self.load_artifacts()
            if self.model is None:
                raise RuntimeError("Prediction model is not trained or loaded.")
                
        df_feat = self.prepare_input_features(input_dict)
        feature_vector = df_feat[self.feature_cols].copy()
        
        # Linear Regression needs scaling; tree models don't necessarily, but let's check model type
        best_name = self.metadata.get("best_model_name", "Gradient Boosting")
        if "Linear" in best_name and self.scaler is not None:
            X_in = self.scaler.transform(feature_vector)
        else:
            X_in = feature_vector
            
        raw_pred = float(self.model.predict(X_in)[0])
        pred_rating = round(float(np.clip(raw_pred, 45.0, 99.0)), 1)
        
        # 95% confidence interval using model RMSE
        margin = round(1.96 * self.rmse, 1)
        conf_lower = round(max(45.0, pred_rating - margin), 1)
        conf_upper = round(min(99.0, pred_rating + margin), 1)
        
        # Potential score calculation
        curr_rating = float(df_feat["player_rating"].iloc[0])
        age = int(df_feat["age"].iloc[0])
        minutes = int(df_feat["minutes_played"].iloc[0])
        
        age_mult = 1.15 if age <= 20 else (1.08 if age <= 23 else (1.02 if age <= 26 else (0.96 if age <= 29 else 0.88)))
        exp_factor = np.clip(minutes / 2000.0, 0.75, 1.05)
        
        raw_pot = (0.40 * pred_rating + 0.25 * curr_rating + 0.20 * (pred_rating * age_mult) + 0.15 * (curr_rating * exp_factor))
        potential_score = round(float(np.clip(raw_pot, 45.0, 99.0)), 1)
        
        # Prospect Tier
        if potential_score >= 88.0:
            tier = "Elite Prospect"
        elif potential_score >= 80.0:
            tier = "High Potential"
        elif potential_score >= 72.0:
            tier = "Promising"
        elif potential_score >= 64.0:
            tier = "Development Player"
        else:
            tier = "Low Priority"
            
        # Top Contributing Factors (Explainable AI)
        top_importances = self.metadata.get("model_comparison", {}).get(best_name, {}).get("top_features", [])
        factors = []
        for item in top_importances[:6]:
            feat = item["feature"]
            imp = item["importance"]
            val = df_feat[feat].iloc[0] if feat in df_feat.columns else 0
            
            # Format friendly name
            friendly_name = feat.replace("_", " ").title()
            direction = "positive" if val >= 50 or (feat == "age" and val <= 24) else "neutral"
            factors.append({
                "feature": feat,
                "display_name": friendly_name,
                "value": round(float(val), 2) if isinstance(val, (int, float, np.number)) else str(val),
                "importance_pct": imp,
                "direction": direction
            })
            
        return {
            "predicted_rating": pred_rating,
            "confidence_lower": conf_lower,
            "confidence_upper": conf_upper,
            "confidence_margin": margin,
            "potential_score": potential_score,
            "potential_tier": tier,
            "current_rating": curr_rating,
            "age": age,
            "attacking_score": float(df_feat["attacking_score"].iloc[0]),
            "defensive_score": float(df_feat["defensive_score"].iloc[0]),
            "passing_score": float(df_feat["passing_score"].iloc[0]),
            "progression_score": float(df_feat["progression_score"].iloc[0]),
            "overall_performance_score": float(df_feat["overall_performance_score"].iloc[0]),
            "top_factors": factors
        }
