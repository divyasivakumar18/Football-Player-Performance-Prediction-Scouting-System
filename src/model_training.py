"""
Machine Learning Model Training, Hyperparameter Tuning, Evaluation, and Player Clustering.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import xgboost as xgb

SEED = 42

MODEL_FEATURES = [
    "age", "appearances", "minutes_played", "starts", "goals", "assists",
    "shots", "shots_on_target", "passes", "pass_accuracy", "key_passes",
    "chances_created", "tackles", "interceptions", "clearances", "blocks",
    "recoveries", "aerial_duels_won", "defensive_actions", "dribbles",
    "successful_dribbles", "progressive_carries", "progressive_passes",
    "distance_covered", "sprint_count", "fouls", "yellow_cards", "saves",
    "save_percentage", "clean_sheets", "goals_conceded", "player_rating",
    "previous_season_rating", "injury_days", "goals_per_90", "assists_per_90",
    "key_passes_per_90", "tackles_per_90", "interceptions_per_90",
    "progressive_passes_per_90", "attacking_score", "defensive_score",
    "passing_score", "progression_score", "overall_performance_score"
]

CLUSTERING_FEATURES = [
    "goals_per_90", "assists_per_90", "shots_on_target_per_90", "key_passes_per_90",
    "tackles_per_90", "interceptions_per_90", "clearances_per_90", "recoveries_per_90",
    "progressive_carries_per_90", "progressive_passes_per_90", "successful_dribbles_per_90",
    "attacking_score", "defensive_score", "passing_score", "progression_score"
]

ARCHETYPE_NAMES = {
    0: "Creative Playmakers",
    1: "Goal Scorers & Finishers",
    2: "Ball-Winning Engines",
    3: "Rock-Solid Defenders",
    4: "Balanced All-Rounders"
}


def prepare_training_data(df: pd.DataFrame):
    """
    Performs chronological train/test split.
    Train: Seasons 2021-2022, 2022-2023, 2023-2024 (~75-80%)
    Test: Season 2024-2025 (~20-25%)
    """
    target_col = "future_performance_rating"
    
    # Filter available features
    feature_cols = [c for c in MODEL_FEATURES if c in df.columns]
    
    # Chronological split
    test_season = "2024-2025"
    train_df = df[df["season"] != test_season]
    test_df = df[df["season"] == test_season]
    
    # Fallback to 80/20 if single season
    if len(test_df) == 0:
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].values
    
    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].values
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        "X_train": X_train,
        "X_train_scaled": X_train_scaled,
        "y_train": y_train,
        "X_test": X_test,
        "X_test_scaled": X_test_scaled,
        "y_test": y_test,
        "feature_cols": feature_cols,
        "scaler": scaler
    }


def train_and_evaluate_models(data_dict: dict) -> tuple[dict, object, str]:
    """Trains Linear Regression, Random Forest, Gradient Boosting, and XGBoost models."""
    X_train = data_dict["X_train"]
    X_train_scaled = data_dict["X_train_scaled"]
    y_train = data_dict["y_train"]
    X_test = data_dict["X_test"]
    X_test_scaled = data_dict["X_test_scaled"]
    y_test = data_dict["y_test"]
    feature_cols = data_dict["feature_cols"]
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=250, max_depth=12, min_samples_split=4, random_state=SEED, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.06, max_depth=5, random_state=SEED
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=250, learning_rate=0.05, max_depth=5, subsample=0.85,
            colsample_bytree=0.85, random_state=SEED, n_jobs=-1
        )
    }
    
    results = {}
    best_r2 = -float("inf")
    best_model_name = None
    best_model = None
    
    for name, model in models.items():
        # Train
        if name == "Linear Regression":
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
        mae = float(mean_absolute_error(y_test, preds))
        mse = float(mean_squared_error(y_test, preds))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test, preds))
        
        # Calculate feature importances
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_.tolist()
            feat_imp = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
            top_features = [{"feature": k, "importance": round(v * 100, 2)} for k, v in feat_imp[:12]]
        elif hasattr(model, "coef_"):
            coefs = np.abs(model.coef_).tolist()
            total = sum(coefs) if sum(coefs) > 0 else 1
            feat_imp = sorted(zip(feature_cols, [c / total for c in coefs]), key=lambda x: x[1], reverse=True)
            top_features = [{"feature": k, "importance": round(v * 100, 2)} for k, v in feat_imp[:12]]
        else:
            top_features = []
            
        results[name] = {
            "MAE": round(mae, 3),
            "MSE": round(mse, 3),
            "RMSE": round(rmse, 3),
            "R2": round(r2, 4),
            "top_features": top_features,
            "sample_predictions": [round(float(p), 2) for p in preds[:10]],
            "sample_actuals": [round(float(a), 2) for a in y_test[:10]]
        }
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_model = model
            
    return results, best_model, best_model_name


def train_player_clustering(df: pd.DataFrame, n_clusters: int = 5) -> tuple[KMeans, StandardScaler, PCA, dict]:
    """
    Fits K-Means clustering and PCA for player archetype identification.
    """
    feature_cols = [c for c in CLUSTERING_FEATURES if c in df.columns]
    X_cluster = df[feature_cols].copy().fillna(0)
    
    cluster_scaler = StandardScaler()
    X_scaled = cluster_scaler.fit_transform(X_cluster)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=15)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # 2D & 3D PCA for dashboard visualization
    pca_2d = PCA(n_components=2, random_state=SEED)
    pca_coords_2d = pca_2d.fit_transform(X_scaled)
    
    pca_3d = PCA(n_components=3, random_state=SEED)
    pca_coords_3d = pca_3d.fit_transform(X_scaled)
    
    sil_score = float(silhouette_score(X_scaled, cluster_labels))
    
    # Analyze cluster centroids to assign meaningful archetype labels
    centroids = kmeans.cluster_centers_
    cluster_info = {}
    
    # Map cluster index to archetype based on dominant attributes
    for c_id in range(n_clusters):
        c_mask = cluster_labels == c_id
        cluster_df = df.iloc[c_mask]
        
        avg_atk = float(cluster_df["attacking_score"].mean()) if "attacking_score" in cluster_df else 0
        avg_def = float(cluster_df["defensive_score"].mean()) if "defensive_score" in cluster_df else 0
        avg_pass = float(cluster_df["passing_score"].mean()) if "passing_score" in cluster_df else 0
        avg_prog = float(cluster_df["progression_score"].mean()) if "progression_score" in cluster_df else 0
        
        # Categorize
        if avg_atk > 65 and avg_atk >= avg_def:
            name = "Goal Scorers & Attack Finishers"
        elif avg_def > 60 and avg_def > avg_pass:
            name = "Rock-Solid Defenders"
        elif avg_pass > 65 and avg_prog > 55:
            name = "Creative Playmakers & Creators"
        elif avg_def > 45 and avg_pass > 50:
            name = "Ball-Winning Engine & Anchors"
        else:
            name = "Dynamic All-Rounders"
            
        cluster_info[c_id] = {
            "name": name,
            "count": int(c_mask.sum()),
            "avg_attacking": round(avg_atk, 1),
            "avg_defensive": round(avg_def, 1),
            "avg_passing": round(avg_pass, 1),
            "avg_progression": round(avg_prog, 1),
            "top_players": cluster_df.nlargest(3, "player_rating")["player_name"].tolist() if "player_name" in cluster_df else []
        }
        
    metadata = {
        "n_clusters": n_clusters,
        "silhouette_score": round(sil_score, 3),
        "explained_variance_ratio_2d": [round(float(v), 3) for v in pca_2d.explained_variance_ratio_],
        "cluster_profiles": cluster_info,
        "feature_cols": feature_cols
    }
    
    return kmeans, cluster_scaler, pca_2d, metadata


def run_full_training_pipeline(processed_csv_path: str, models_dir: str):
    """Orchestrates end-to-end ML and clustering pipeline."""
    os.makedirs(models_dir, exist_ok=True)
    df = pd.read_csv(processed_csv_path)
    
    print("1. Preparing Chronological Train/Test Datasets...")
    data_dict = prepare_training_data(df)
    
    print("2. Training Regression Models (Linear, RF, GBDT, XGBoost)...")
    eval_results, best_model, best_model_name = train_and_evaluate_models(data_dict)
    
    print(f"--> Best Model: {best_model_name} (R² = {eval_results[best_model_name]['R2']}, RMSE = {eval_results[best_model_name]['RMSE']})")
    
    print("3. Training K-Means Player Clustering & PCA...")
    kmeans_model, cluster_scaler, pca_model, cluster_meta = train_player_clustering(df, n_clusters=5)
    
    # Save artifacts
    print("4. Serializing Models and Metadata...")
    joblib.dump(best_model, os.path.join(models_dir, "performance_model.pkl"))
    joblib.dump(data_dict["scaler"], os.path.join(models_dir, "scaler.pkl"))
    joblib.dump(kmeans_model, os.path.join(models_dir, "clustering_model.pkl"))
    joblib.dump(cluster_scaler, os.path.join(models_dir, "cluster_scaler.pkl"))
    joblib.dump(pca_model, os.path.join(models_dir, "pca_model.pkl"))
    
    metadata = {
        "best_model_name": best_model_name,
        "model_comparison": eval_results,
        "feature_cols": data_dict["feature_cols"],
        "clustering_metadata": cluster_meta
    }
    
    with open(os.path.join(models_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"All models and metadata successfully saved to: {models_dir}")
    return metadata


if __name__ == "__main__":
    import sys
    if os.path.dirname(os.path.dirname(__file__)) not in sys.path:
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        
    proc_csv = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "players_processed.csv")
    mdl_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    meta = run_full_training_pipeline(proc_csv, mdl_dir)
    print("Training Pipeline Finished Successfully!")
