"""
Scouting Recommendation Engine, Position Recommender, Hidden Gem Finder,
and Percentile Strength/Weakness Benchmarking.
"""

import numpy as np
import pandas as pd


def calculate_scouting_match_score(
    player_row: pd.Series,
    target_pos: str = None,
    min_age: int = 16,
    max_age: int = 35,
    min_rating: float = 60.0,
    req_goals: int = 0,
    req_assists: int = 0,
    min_pass_acc: float = 0.0,
    min_def_score: float = 0.0,
    max_budget: float = 200.0
) -> float:
    """
    Computes dynamic scouting match score (0-100) based on user filter criteria.
    """
    # 1. Position match (20 pts)
    pos = player_row.get("position", "")
    if target_pos is None or target_pos == "All" or pos == target_pos:
        pos_match = 20.0
    else:
        pos_match = 5.0
        
    # 2. Performance match (30 pts)
    curr_rating = float(player_row.get("player_rating", 70.0))
    if curr_rating >= min_rating:
        perf_match = 20.0 + min(10.0, (curr_rating - min_rating) * 0.5)
    else:
        perf_match = max(0.0, 20.0 - (min_rating - curr_rating) * 2.0)
        
    # 3. Age profile match (15 pts)
    age = int(player_row.get("age", 25))
    if min_age <= age <= max_age:
        # Younger within the window gets a slight bonus
        age_match = 15.0 - ((age - min_age) / max(1, max_age - min_age)) * 3.0
    else:
        dist = min(abs(age - min_age), abs(age - max_age))
        age_match = max(0.0, 15.0 - dist * 3.0)
        
    # 4. Statistical requirements match (15 pts)
    p_goals = player_row.get("goals", 0)
    p_assists = player_row.get("assists", 0)
    p_pass = player_row.get("pass_accuracy", 70.0)
    p_def = player_row.get("defensive_score", 50.0)
    
    stat_pts = 0.0
    stat_pts += 4.0 if p_goals >= req_goals else max(0.0, 4.0 * (p_goals / max(1, req_goals)))
    stat_pts += 4.0 if p_assists >= req_assists else max(0.0, 4.0 * (p_assists / max(1, req_assists)))
    stat_pts += 3.5 if p_pass >= min_pass_acc else max(0.0, 3.5 * (p_pass / max(1.0, min_pass_acc)))
    stat_pts += 3.5 if p_def >= min_def_score else max(0.0, 3.5 * (p_def / max(1.0, min_def_score)))
    stat_match = min(15.0, stat_pts)
    
    # 5. Potential bonus (10 pts)
    pot = float(player_row.get("potential_score", curr_rating))
    pot_match = (pot / 100.0) * 10.0
    
    # 6. Budget efficiency match (10 pts)
    mv = float(player_row.get("market_value", 20.0))
    if mv <= max_budget:
        budget_match = 7.0 + min(3.0, (max_budget - mv) / max(1.0, max_budget) * 3.0)
    else:
        budget_match = max(0.0, 7.0 - (mv - max_budget) * 0.4)
        
    total_score = pos_match + perf_match + age_match + stat_match + pot_match + budget_match
    return round(float(np.clip(total_score, 0.0, 100.0)), 1)


def recommend_scouted_players(
    df: pd.DataFrame,
    target_pos: str = "All",
    min_age: int = 16,
    max_age: int = 35,
    min_rating: float = 60.0,
    req_goals: int = 0,
    req_assists: int = 0,
    min_pass_acc: float = 0.0,
    min_def_score: float = 0.0,
    max_budget: float = 200.0,
    top_n: int = 25
) -> pd.DataFrame:
    """
    Ranks all players matching scouting requirements by their Scouting Match Score.
    """
    res_df = df.copy()
    
    # Hard base filtering for reasonable scope
    if target_pos != "All":
        res_df = res_df[res_df["position"] == target_pos]
        
    res_df = res_df[
        (res_df["age"] >= min_age) &
        (res_df["age"] <= max_age) &
        (res_df["player_rating"] >= max(45.0, min_rating - 8.0)) &
        (res_df["market_value"] <= max_budget * 1.3)
    ]
    
    if len(res_df) == 0:
        return pd.DataFrame()
        
    scores = []
    for _, row in res_df.iterrows():
        sc = calculate_scouting_match_score(
            row, target_pos, min_age, max_age, min_rating,
            req_goals, req_assists, min_pass_acc, min_def_score, max_budget
        )
        scores.append(sc)
        
    res_df["scouting_match_score"] = scores
    ranked_df = res_df.sort_values(by=["scouting_match_score", "potential_score", "player_rating"], ascending=False)
    
    return ranked_df.head(top_n)


def recommend_alternative_position(player_row: pd.Series) -> dict:
    """
    Analyzes player tactical metrics to suggest alternative roles and tactical flexibility.
    """
    pos = player_row.get("position", "MF")
    d_pos = player_row.get("detailed_position", "CM")
    atk = float(player_row.get("attacking_score", 50.0))
    defense = float(player_row.get("defensive_score", 50.0))
    passing = float(player_row.get("passing_score", 50.0))
    prog = float(player_row.get("progression_score", 50.0))
    duels = float(player_row.get("aerial_duels_won", 10.0))
    
    alt_pos = None
    reason = ""
    
    if pos == "MF":
        if atk >= 70 and prog >= 65:
            alt_pos = "CAM / Second Striker"
            reason = "Exceptional chance creation, final third dribbling, and attacking presence."
        elif defense >= 70 and duels >= 25:
            alt_pos = "CDM / Ball-Winning Anchor"
            reason = "Dominant tackling, interception volume, and physical duel recovery."
        elif prog >= 72 and atk >= 62:
            alt_pos = "Winger (LW/RW)"
            reason = "Explosive progressive carrying and 1v1 dribble efficiency."
    elif pos == "FW":
        if passing >= 68 and prog >= 65:
            alt_pos = "CAM / Creative Playmaker"
            reason = "Superior vision, key passing ability, and progressive link-up play."
        elif d_pos == "ST" and prog >= 70:
            alt_pos = "Inverted Winger"
            reason = "High ball carrying output and dynamic wide acceleration."
    elif pos == "DF":
        if d_pos in ["LB", "RB"] and prog >= 65 and atk >= 55:
            alt_pos = "Attacking Wing-Back (LWB/RWB) / Wide Midfielder"
            reason = "High offensive overlap output and progressive delivery down the flank."
        elif d_pos == "CB" and passing >= 68:
            alt_pos = "Deep-Lying Midfielder (CDM)"
            reason = "Calm ball progression, high pass accuracy, and defensive security."
    elif pos == "GK":
        if passing >= 65:
            alt_pos = "Sweeper-Keeper"
            reason = "High passing accuracy and proactive distribution from the back."
            
    if alt_pos is None:
        alt_pos = f"Primary {d_pos} Specialist"
        reason = "Stats are optimally aligned with current tactical role."
        
    return {
        "primary_position": f"{pos} ({d_pos})",
        "recommended_alternative": alt_pos,
        "tactical_reason": reason
    }


def compute_strength_weaknesses(player_row: pd.Series, position_df: pd.DataFrame) -> dict:
    """
    Computes position-specific percentile rankings and classifies strengths and weaknesses.
    """
    metrics_to_eval = [
        ("Finishing / Goals", "goals_per_90"),
        ("Creativity / Assists", "assists_per_90"),
        ("Key Passes & Chances", "key_passes_per_90"),
        ("Passing Accuracy", "pass_accuracy"),
        ("Ball Progression", "progressive_passes_per_90"),
        ("1v1 Dribbling", "successful_dribbles_per_90"),
        ("Tackles & Interceptions", "tackles_per_90"),
        ("Clearances & Aerial Duels", "clearances_per_90"),
        ("Defensive Recoveries", "recoveries_per_90"),
        ("Work Rate & Distance", "distance_covered")
    ]
    
    strengths = []
    weaknesses = []
    percentiles = {}
    
    for label, col in metrics_to_eval:
        if col in position_df.columns and col in player_row:
            p_val = player_row[col]
            series = position_df[col].dropna()
            if len(series) > 0:
                pct = round(float((series < p_val).mean() * 100), 1)
                percentiles[label] = pct
                if pct >= 70.0:
                    strengths.append(f"{label} ({pct}th percentile)")
                elif pct <= 30.0:
                    weaknesses.append(f"{label} ({pct}th percentile)")
                    
    return {
        "strengths": strengths[:4] if strengths else ["Consistent Role Execution"],
        "weaknesses": weaknesses[:3] if weaknesses else ["No Severe Weaknesses"],
        "percentile_dict": percentiles
    }


def find_hidden_gems(df: pd.DataFrame, max_age: int = 23, max_mv: float = 35.0, min_pot: float = 78.0) -> pd.DataFrame:
    """
    Filters for high-potential, high-performing undervalued young prospects.
    """
    gems = df[
        (df["age"] <= max_age) &
        (df["market_value"] <= max_mv) &
        (df["potential_score"] >= min_pot) &
        (df["future_performance_rating"] >= 76.0)
    ].copy()
    
    gems["gem_value_ratio"] = (gems["potential_score"] / np.maximum(gems["market_value"], 1.0)).round(2)
    return gems.sort_values(by=["gem_value_ratio", "potential_score"], ascending=False)
