"""
Jupyter Notebooks Generator for Football IQ Project.
Creates 5 fully structured, professional notebooks in the notebooks/ folder.
"""

import os
import json

NOTEBOOKS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "notebooks")
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.11"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


import uuid

def md_cell(text):
    return {
        "id": str(uuid.uuid4())[:8],
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    }


def code_cell(code):
    return {
        "id": str(uuid.uuid4())[:8],
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")]
    }


def create_nb1():
    cells = [
        md_cell("# 01. Data Cleaning & Validation Pipeline\n\n**Football IQ - Player Performance Prediction & Scouting System**\n\nThis notebook demonstrates the end-to-end data cleaning, handling of missing values, deduplication, sanity checking, and IQR outlier analysis."),
        code_cell("""import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load raw player performance data
raw_path = os.path.join("..", "data", "raw", "players.csv")
df_raw = pd.read_csv(raw_path)
print(f"Raw dataset shape: {df_raw.shape}")
df_raw.head()"""),
        md_cell("### 1. Data Inspection & Missing Value Audit"),
        code_cell("""# Checking data types and missing value counts
print("Missing values per column:")
print(df_raw.isnull().sum()[df_raw.isnull().sum() > 0])
print("\\nDataset Info:")
df_raw.info()"""),
        md_cell("### 2. Run Cleaning Pipeline from `src.data_processing`"),
        code_cell("""import sys
sys.path.append("..")
from src.data_processing import clean_player_data

df_clean, report = clean_player_data(df_raw)
print("Data Quality Report:")
for k, v in report.items():
    print(f" - {k}: {v}")"""),
        md_cell("### 3. Outlier Analysis & Distribution Checks"),
        code_cell("""# Visualizing distributions of key metrics
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
sns.boxplot(y=df_clean["goals"], ax=axes[0], color="skyblue")
axes[0].set_title("Goals Distribution (IQR Analysis)")

sns.boxplot(y=df_clean["pass_accuracy"], ax=axes[1], color="lightgreen")
axes[1].set_title("Pass Accuracy (%) Distribution")

sns.boxplot(y=df_clean["market_value"], ax=axes[2], color="salmon")
axes[2].set_title("Market Value (€M) Distribution")
plt.tight_layout()
plt.show()""")
    ]
    path = os.path.join(NOTEBOOKS_DIR, "01_data_cleaning.ipynb")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_notebook(cells), f, indent=2)
    print(f"Created: {path}")


def create_nb2():
    cells = [
        md_cell("# 02. Exploratory Data Analysis (EDA)\n\n**Football IQ - Player Performance Prediction & Scouting System**\n\nComprehensive exploration of player performance across positions, leagues, age trajectories, and club contributions."),
        code_cell("""import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load processed player data
proc_path = os.path.join("..", "data", "processed", "players_processed.csv")
df = pd.read_csv(proc_path)
print(f"Processed dataset shape: {df.shape}")
df.head()"""),
        md_cell("### 1. Positional Performance Benchmarks"),
        code_cell("""# Average ratings and key metrics by position
pos_summary = df.groupby("position")[["player_rating", "goals", "assists", "pass_accuracy", "tackles", "defensive_score", "attacking_score"]].mean().round(2)
print("Average Metrics by Position:")
display(pos_summary)"""),
        md_cell("### 2. Age vs Performance Curve"),
        code_cell("""plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x="age", y="player_rating", hue="position", marker="o", errorbar=None)
plt.title("Player Rating vs Age by Position")
plt.xlabel("Age")
plt.ylabel("Average Rating")
plt.grid(True, alpha=0.3)
plt.show()"""),
        md_cell("### 3. Attacking Contribution (Goals vs Assists)"),
        code_cell("""plt.figure(figsize=(10, 6))
fw_mf = df[df["position"].isin(["FW", "MF"])]
sns.scatterplot(data=fw_mf, x="goals", y="assists", hue="position", size="market_value", sizes=(30, 300), alpha=0.7)
plt.title("Goals vs Assists for Forwards and Midfielders")
plt.xlabel("Goals")
plt.ylabel("Assists")
plt.show()"""),
        md_cell("### 4. Correlation Heatmap"),
        code_cell("""corr_cols = ["age", "minutes_played", "goals", "assists", "pass_accuracy", "key_passes", "tackles", "interceptions", "player_rating", "future_performance_rating", "market_value"]
plt.figure(figsize=(12, 8))
sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Heatmap of Key Football Metrics")
plt.show()""")
    ]
    path = os.path.join(NOTEBOOKS_DIR, "02_eda.ipynb")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_notebook(cells), f, indent=2)
    print(f"Created: {path}")


def create_nb3():
    cells = [
        md_cell("# 03. Feature Engineering & Domain Scoring\n\n**Football IQ - Player Performance Prediction & Scouting System**\n\nImplementation and validation of per-90 metrics, tactical composite scores (Attacking, Defensive, Passing, Progression), and potential formulas."),
        code_cell("""import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append("..")
from src.feature_engineering import (
    compute_per_90_metrics,
    calculate_tactical_scores,
    calculate_overall_performance_score,
    calculate_potential_score
)

# Load clean data
clean_path = os.path.join("..", "data", "processed", "players_processed.csv")
df = pd.read_csv(clean_path)
df.head()"""),
        md_cell("### 1. Tactical Scores Analysis"),
        code_cell("""fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sns.histplot(df["attacking_score"], bins=30, kde=True, ax=axes[0, 0], color="crimson")
axes[0, 0].set_title("Attacking Score Distribution (0 - 100)")

sns.histplot(df["defensive_score"], bins=30, kde=True, ax=axes[0, 1], color="royalblue")
axes[0, 1].set_title("Defensive Score Distribution (0 - 100)")

sns.histplot(df["passing_score"], bins=30, kde=True, ax=axes[1, 0], color="forestgreen")
axes[1, 0].set_title("Passing Score Distribution (0 - 100)")

sns.histplot(df["potential_score"], bins=30, kde=True, ax=axes[1, 1], color="gold")
axes[1, 1].set_title("Potential Score Distribution (0 - 100)")

plt.tight_layout()
plt.show()"""),
        md_cell("### 2. Potential Tier Categorization"),
        code_cell("""print("Prospect Count by Potential Tier:")
print(df["potential_tier"].value_counts())

plt.figure(figsize=(8, 4))
sns.countplot(data=df, x="potential_tier", order=["Elite Prospect", "High Potential", "Promising", "Development Player", "Low Priority"], palette="viridis")
plt.title("Distribution of Players Across Potential Tiers")
plt.xticks(rotation=20)
plt.show()""")
    ]
    path = os.path.join(NOTEBOOKS_DIR, "03_feature_engineering.ipynb")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_notebook(cells), f, indent=2)
    print(f"Created: {path}")


def create_nb4():
    cells = [
        md_cell("# 04. Machine Learning Model Training & Evaluation\n\n**Football IQ - Player Performance Prediction & Scouting System**\n\nTraining, chronological validation, hyperparameter tuning, and metric comparison across Linear Regression, Random Forest, Gradient Boosting, and XGBoost models."),
        code_cell("""import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append("..")
from src.model_training import prepare_training_data, train_and_evaluate_models

# Load dataset
proc_path = os.path.join("..", "data", "processed", "players_processed.csv")
df = pd.read_csv(proc_path)

# Prepare chronological train/test split
data_dict = prepare_training_data(df)
print(f"Training Samples (Seasons 2021-24): {len(data_dict['X_train'])}")
print(f"Testing Samples (Season 2024-25): {len(data_dict['X_test'])}")
print(f"Number of Features: {len(data_dict['feature_cols'])}")"""),
        md_cell("### 1. Train and Benchmark All Regressors"),
        code_cell("""results, best_model, best_model_name = train_and_evaluate_models(data_dict)

print(f"--- BEST MODEL: {best_model_name} ---\\n")
metrics_table = []
for name, m in results.items():
    metrics_table.append({
        "Model": name,
        "MAE": m["MAE"],
        "MSE": m["MSE"],
        "RMSE": m["RMSE"],
        "R2": m["R2"]
    })
metrics_df = pd.DataFrame(metrics_table).sort_values("R2", ascending=False)
display(metrics_df)"""),
        md_cell("### 2. Actual vs. Predicted Performance"),
        code_cell("""# Model predictions on test set
X_test = data_dict["X_test"]
y_test = data_dict["y_test"]
preds = best_model.predict(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, preds, alpha=0.5, color="teal")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
plt.xlabel("Actual Future Rating")
plt.ylabel("Predicted Future Rating")
plt.title(f"Actual vs. Predicted Future Rating ({best_model_name})")
plt.grid(True, alpha=0.3)
plt.show()"""),
        md_cell("### 3. Feature Importance Analysis"),
        code_cell("""feat_imp = results[best_model_name]["top_features"]
imp_df = pd.DataFrame(feat_imp).sort_values("importance", ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(imp_df["feature"], imp_df["importance"], color="steelblue")
plt.xlabel("Relative Importance (%)")
plt.title(f"Top 12 Predictive Features ({best_model_name})")
plt.tight_layout()
plt.show()""")
    ]
    path = os.path.join(NOTEBOOKS_DIR, "04_model_training.ipynb")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_notebook(cells), f, indent=2)
    print(f"Created: {path}")


def create_nb5():
    cells = [
        md_cell("# 05. Unsupervised Player Clustering & Archetypes\n\n**Football IQ - Player Performance Prediction & Scouting System**\n\nK-Means clustering, Elbow method, Silhouette analysis, and 2D/3D PCA dimensionality reduction to discover tactical player archetypes."),
        code_cell("""import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

import sys
sys.path.append("..")
from src.model_training import train_player_clustering, CLUSTERING_FEATURES, ARCHETYPE_NAMES

# Load processed data
df = pd.read_csv(os.path.join("..", "data", "processed", "players_processed.csv"))
kmeans, c_scaler, pca_2d, meta = train_player_clustering(df, n_clusters=5)

print(f"Silhouette Score: {meta['silhouette_score']}")
print("Cluster Profiles:")
for c_id, prof in meta['cluster_profiles'].items():
    print(f" - Cluster {c_id} ({prof['name']}): {prof['count']} players")"""),
        md_cell("### 1. Elbow Method & Inertia Plot"),
        code_cell("""feat_cols = [c for c in CLUSTERING_FEATURES if c in df.columns]
X_scaled = c_scaler.transform(df[feat_cols].fillna(0))

inertias = []
k_range = range(2, 10)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(k_range, inertias, "bo-")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (Sum of Squared Distances)")
plt.title("Elbow Method for Optimal k")
plt.grid(True)
plt.show()"""),
        md_cell("### 2. 2D PCA Cluster Scatter Plot"),
        code_cell("""pca_coords = pca_2d.transform(X_scaled)
df_plot = df.copy()
df_plot["cluster_id"] = kmeans.predict(X_scaled)
df_plot["archetype"] = df_plot["cluster_id"].map(ARCHETYPE_NAMES)
df_plot["PCA_1"] = pca_coords[:, 0]
df_plot["PCA_2"] = pca_coords[:, 1]

plt.figure(figsize=(10, 7))
sns.scatterplot(data=df_plot, x="PCA_1", y="PCA_2", hue="archetype", palette="Set1", alpha=0.7)
plt.title("2D PCA Projection of Tactical Player Clusters")
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()""")
    ]
    path = os.path.join(NOTEBOOKS_DIR, "05_player_clustering.ipynb")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_notebook(cells), f, indent=2)
    print(f"Created: {path}")


if __name__ == "__main__":
    create_nb1()
    create_nb2()
    create_nb3()
    create_nb4()
    create_nb5()
    print("All 5 Jupyter notebooks generated successfully!")
