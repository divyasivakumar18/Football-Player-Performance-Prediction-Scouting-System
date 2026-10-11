"""
Realistic Football Player Dataset Generator.
Generates multi-season player performance data across top European leagues with authentic positional statistics.
"""

import os
import random
import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

LEAGUES_AND_CLUBS = {
    "Premier League": ["Manchester City", "Arsenal", "Liverpool", "Aston Villa", "Tottenham", "Chelsea", "Manchester United", "Newcastle United", "Brighton", "West Ham"],
    "La Liga": ["Real Madrid", "Barcelona", "Atletico Madrid", "Real Sociedad", "Athletic Club", "Girona", "Real Betis", "Villarreal", "Sevilla", "Valencia"],
    "Serie A": ["Inter Milan", "AC Milan", "Juventus", "Atalanta", "AS Roma", "Napoli", "Lazio", "Fiorentina", "Bologna", "Torino"],
    "Bundesliga": ["Bayern Munich", "Bayer Leverkusen", "Borussia Dortmund", "RB Leipzig", "Eintracht Frankfurt", "VfB Stuttgart", "Wolfsburg", "Freiburg", "Hoffenheim"],
    "Ligue 1": ["Paris Saint-Germain", "Monaco", "Marseille", "Lille", "Lyon", "Nice", "Lens", "Rennes", "Toulouse"]
}

NATIONALITIES = [
    "England", "France", "Spain", "Germany", "Brazil", "Argentina", 
    "Portugal", "Netherlands", "Italy", "Belgium", "Norway", "Uruguay",
    "Croatia", "Denmark", "Nigeria", "Japan", "Colombia", "Sweden", "Morocco", "Senegal"
]

FIRST_NAMES = [
    "Alexander", "Marcus", "Lucas", "Julian", "Gabriel", "Mateo", "Liam", "Bruno", "Florian",
    "Dominik", "Kylian", "Erling", "Jude", "Bukayo", "Mohamed", "Vinicius", "Rodri", "Pedri",
    "William", "Virgil", "Jamal", "Lamine", "Kobbie", "Arda", "Eduardo", "Federico", "Declan",
    "Cole", "Phil", "Martin", "Joshua", "Rafael", "Victor", "Lautaro", "Khvicha", "Aurélien",
    "Theo", "Alphonso", "Jeremie", "Achraf", "Josko", "Ruben", "Alessandro", "Milos", "Kenan",
    "Takefusa", "Enzo", "Gavi", "Alejandro", "Endrick", "Evan", "Castello", "Benjamin", "Warren"
]

LAST_NAMES = [
    "Silva", "Santos", "Garcia", "Fernandez", "Smith", "Müller", "Schneider", "Davies",
    "Hernandez", "Bellingham", "Saka", "Haaland", "Mbappé", "Salah", "Junior", "Guler",
    "Mainoo", "Yamal", "Musiala", "Wirtz", "Rice", "Foden", "Palmer", "Odegaard", "Saliba",
    "Van Dijk", "Alisson", "Ederson", "Courtois", "Donnarumma", "Martinez", "Valverde",
    "Camavinga", "Tchouaméni", "Leao", "Osimhen", "Kvaratskhelia", "Bastoni", "Frimpong",
    "Hakimi", "Gvardiol", "Dias", "Alvarez", "Gordon", "Isak", "Guimaraes", "Barella", "Calafiori"
]

DETAILED_POSITIONS = {
    "FW": ["ST", "LW", "RW", "CF"],
    "MF": ["CAM", "CM", "CDM"],
    "DF": ["CB", "LB", "RB"],
    "GK": ["GK"]
}

SEASONS = ["2021-2022", "2022-2023", "2023-2024", "2024-2025"]

REAL_STAR_PLAYERS = [
    {"name": "Erling Haaland", "nationality": "Norway", "position": "FW", "detailed": "ST", "club": "Manchester City", "league": "Premier League", "base_rating": 91, "base_age": 21},
    {"name": "Kylian Mbappé", "nationality": "France", "position": "FW", "detailed": "ST", "club": "Real Madrid", "league": "La Liga", "base_rating": 91, "base_age": 23},
    {"name": "Jude Bellingham", "nationality": "England", "position": "MF", "detailed": "CAM", "club": "Real Madrid", "league": "La Liga", "base_rating": 89, "base_age": 18},
    {"name": "Vinícius Júnior", "nationality": "Brazil", "position": "FW", "detailed": "LW", "club": "Real Madrid", "league": "La Liga", "base_rating": 89, "base_age": 21},
    {"name": "Rodri Hernandez", "nationality": "Spain", "position": "MF", "detailed": "CDM", "club": "Manchester City", "league": "Premier League", "base_rating": 91, "base_age": 25},
    {"name": "Bukayo Saka", "nationality": "England", "position": "FW", "detailed": "RW", "club": "Arsenal", "league": "Premier League", "base_rating": 87, "base_age": 20},
    {"name": "Mohamed Salah", "nationality": "Egypt", "position": "FW", "detailed": "RW", "club": "Liverpool", "league": "Premier League", "base_rating": 89, "base_age": 29},
    {"name": "Florian Wirtz", "nationality": "Germany", "position": "MF", "detailed": "CAM", "club": "Bayer Leverkusen", "league": "Bundesliga", "base_rating": 88, "base_age": 18},
    {"name": "Jamal Musiala", "nationality": "Germany", "position": "MF", "detailed": "CAM", "club": "Bayern Munich", "league": "Bundesliga", "base_rating": 88, "base_age": 18},
    {"name": "Lamine Yamal", "nationality": "Spain", "position": "FW", "detailed": "RW", "club": "Barcelona", "league": "La Liga", "base_rating": 84, "base_age": 16},
    {"name": "Cole Palmer", "nationality": "England", "position": "MF", "detailed": "CAM", "club": "Chelsea", "league": "Premier League", "base_rating": 85, "base_age": 19},
    {"name": "Phil Foden", "nationality": "England", "position": "MF", "detailed": "CAM", "club": "Manchester City", "league": "Premier League", "base_rating": 88, "base_age": 21},
    {"name": "Martin Odegaard", "nationality": "Norway", "position": "MF", "detailed": "CAM", "club": "Arsenal", "league": "Premier League", "base_rating": 87, "base_age": 23},
    {"name": "Declan Rice", "nationality": "England", "position": "MF", "detailed": "CDM", "club": "Arsenal", "league": "Premier League", "base_rating": 87, "base_age": 22},
    {"name": "William Saliba", "nationality": "France", "position": "DF", "detailed": "CB", "club": "Arsenal", "league": "Premier League", "base_rating": 87, "base_age": 20},
    {"name": "Virgil van Dijk", "nationality": "Netherlands", "position": "DF", "detailed": "CB", "club": "Liverpool", "league": "Premier League", "base_rating": 89, "base_age": 30},
    {"name": "Ruben Dias", "nationality": "Portugal", "position": "DF", "detailed": "CB", "club": "Manchester City", "league": "Premier League", "base_rating": 88, "base_age": 24},
    {"name": "Josko Gvardiol", "nationality": "Croatia", "position": "DF", "detailed": "LB", "club": "Manchester City", "league": "Premier League", "base_rating": 84, "base_age": 19},
    {"name": "Alphonso Davies", "nationality": "Canada", "position": "DF", "detailed": "LB", "club": "Bayern Munich", "league": "Bundesliga", "base_rating": 84, "base_age": 21},
    {"name": "Jeremie Frimpong", "nationality": "Netherlands", "position": "DF", "detailed": "RB", "club": "Bayer Leverkusen", "league": "Bundesliga", "base_rating": 84, "base_age": 21},
    {"name": "Thibaut Courtois", "nationality": "Belgium", "position": "GK", "detailed": "GK", "club": "Real Madrid", "league": "La Liga", "base_rating": 90, "base_age": 29},
    {"name": "Alisson Becker", "nationality": "Brazil", "position": "GK", "detailed": "GK", "club": "Liverpool", "league": "Premier League", "base_rating": 89, "base_age": 29},
    {"name": "Kobbie Mainoo", "nationality": "England", "position": "MF", "detailed": "CM", "club": "Manchester United", "league": "Premier League", "base_rating": 78, "base_age": 17},
    {"name": "Arda Güler", "nationality": "Turkey", "position": "MF", "detailed": "CAM", "club": "Real Madrid", "league": "La Liga", "base_rating": 78, "base_age": 17},
    {"name": "Alejandro Garnacho", "nationality": "Argentina", "position": "FW", "detailed": "LW", "club": "Manchester United", "league": "Premier League", "base_rating": 79, "base_age": 17},
    {"name": "Endrick", "nationality": "Brazil", "position": "FW", "detailed": "ST", "club": "Real Madrid", "league": "La Liga", "base_rating": 77, "base_age": 16},
    {"name": "Warren Zaïre-Emery", "nationality": "France", "position": "MF", "detailed": "CM", "club": "Paris Saint-Germain", "league": "Ligue 1", "base_rating": 80, "base_age": 16}
]


def generate_player_database(n_synthetic_players=580):
    """Generates a complete multi-season football player performance dataset."""
    player_profiles = []
    
    # 1. Add real star profiles
    p_id = 1001
    for p in REAL_STAR_PLAYERS:
        player_profiles.append({
            "player_id": f"P{p_id}",
            "player_name": p["name"],
            "nationality": p["nationality"],
            "position": p["position"],
            "detailed_position": p["detailed"],
            "club": p["club"],
            "league": p["league"],
            "base_rating": p["base_rating"],
            "base_age": p["base_age"],
            "potential_ceiling": min(95, p["base_rating"] + random.randint(2, 9))
        })
        p_id += 1
        
    # 2. Add realistic synthetic players across leagues and positions
    positions_distribution = ["FW"] * 30 + ["MF"] * 35 + ["DF"] * 25 + ["GK"] * 10
    
    for i in range(n_synthetic_players):
        pos = random.choice(positions_distribution)
        detailed_pos = random.choice(DETAILED_POSITIONS[pos])
        league = random.choice(list(LEAGUES_AND_CLUBS.keys()))
        club = random.choice(LEAGUES_AND_CLUBS[league])
        nat = random.choice(NATIONALITIES)
        first_name = random.choice(FIRST_NAMES)
        last_name = random.choice(LAST_NAMES)
        full_name = f"{first_name} {last_name}"
        
        base_age = random.randint(17, 32)
        # Quality distribution: skewed towards 68-84
        base_rating = int(np.clip(np.random.normal(74, 6.5), 58, 92))
        
        # Potential ceiling depends on age & base rating
        if base_age <= 21:
            growth = random.randint(5, 14)
        elif base_age <= 25:
            growth = random.randint(1, 6)
        else:
            growth = random.randint(-4, 2)
        potential_ceiling = min(94, max(base_rating, base_rating + growth))
        
        player_profiles.append({
            "player_id": f"P{p_id}",
            "player_name": full_name,
            "nationality": nat,
            "position": pos,
            "detailed_position": detailed_pos,
            "club": club,
            "league": league,
            "base_rating": base_rating,
            "base_age": base_age,
            "potential_ceiling": potential_ceiling
        })
        p_id += 1

    records = []
    
    for profile in player_profiles:
        curr_age = profile["base_age"]
        prev_rating = profile["base_rating"] - random.randint(0, 3)
        curr_rating = profile["base_rating"]
        
        for s_idx, season in enumerate(SEASONS):
            # Age progression
            season_age = curr_age + s_idx
            
            # Rating trajectory with career curve
            if season_age <= 23:
                # Young players grow
                growth_rate = np.random.normal(1.8, 0.8)
            elif season_age <= 28:
                # Peak years
                growth_rate = np.random.normal(0.4, 0.6)
            elif season_age <= 32:
                # Slight decline
                growth_rate = np.random.normal(-0.8, 0.7)
            else:
                # Veterans decline faster
                growth_rate = np.random.normal(-1.8, 0.9)
                
            season_rating = float(np.clip(prev_rating + growth_rate, 52.0, 94.5))
            season_rating = round(season_rating, 1)
            
            # Playing time
            starts = int(np.clip(np.random.normal(24, 8), 2, 38))
            sub_appearances = int(np.clip(np.random.normal(6, 4), 0, 15))
            appearances = starts + sub_appearances
            appearances = min(38, appearances)
            minutes_played = int(starts * np.random.uniform(75, 90) + sub_appearances * np.random.uniform(15, 30))
            minutes_played = min(3420, max(120, minutes_played))
            
            # Injury days
            injury_days = int(np.random.exponential(20))
            if injury_days > 120:
                minutes_played = int(minutes_played * 0.6)
                appearances = max(5, int(appearances * 0.6))
                
            # Ratio of play (minutes/3420)
            play_ratio = minutes_played / 3420.0
            pos = profile["position"]
            d_pos = profile["detailed_position"]
            quality_factor = (season_rating - 50) / 45.0  # 0 to 1 scale
            
            # Generate statistics matching football reality
            if pos == "FW":
                goals = int(np.clip(np.random.normal(14 * quality_factor * (play_ratio * 1.2), 4), 0, 36))
                assists = int(np.clip(np.random.normal(7 * quality_factor * play_ratio, 3), 0, 20))
                shots = int(np.clip(goals * 3.5 + np.random.normal(20, 10) * play_ratio, goals, 140))
                shots_on_target = int(np.clip(shots * np.random.uniform(0.40, 0.55), goals, shots))
                passes = int(np.clip(np.random.normal(650 * play_ratio, 150), 100, 1300))
                pass_accuracy = round(float(np.clip(np.random.normal(76 + 8 * quality_factor, 4), 60.0, 90.0)), 1)
                key_passes = int(np.clip(assists * 2.5 + np.random.normal(15, 8) * play_ratio, assists, 85))
                chances_created = int(key_passes + np.random.normal(8, 4) * play_ratio)
                
                tackles = int(np.clip(np.random.normal(18 * play_ratio, 6), 2, 45))
                interceptions = int(np.clip(np.random.normal(10 * play_ratio, 4), 1, 28))
                clearances = int(np.clip(np.random.normal(8 * play_ratio, 4), 1, 25))
                blocks = int(np.clip(np.random.normal(5 * play_ratio, 3), 0, 18))
                recoveries = int(np.clip(np.random.normal(55 * play_ratio, 15), 10, 120))
                aerial_duels_won = int(np.clip(np.random.normal(30 * play_ratio, 15) if d_pos == "ST" else np.random.normal(12 * play_ratio, 6), 2, 90))
                
                dribbles = int(np.clip(np.random.normal(70 * quality_factor * play_ratio, 20), 10, 160))
                successful_dribbles = int(np.clip(dribbles * np.random.uniform(0.48, 0.68), 5, dribbles))
                progressive_carries = int(np.clip(np.random.normal(80 * quality_factor * play_ratio, 25), 15, 180))
                progressive_passes = int(np.clip(np.random.normal(45 * quality_factor * play_ratio, 15), 10, 110))
                ball_losses = int(np.clip(np.random.normal(90 * play_ratio, 20), 20, 180))
                
                saves, save_percentage, clean_sheets, goals_conceded, penalty_saves = 0, 0.0, 0, 0, 0
                
            elif pos == "MF":
                goals = int(np.clip(np.random.normal(5 * quality_factor * play_ratio, 3) if d_pos != "CAM" else np.random.normal(9 * quality_factor * play_ratio, 4), 0, 22))
                assists = int(np.clip(np.random.normal(7 * quality_factor * play_ratio, 3) if d_pos != "CAM" else np.random.normal(11 * quality_factor * play_ratio, 4), 0, 24))
                shots = int(np.clip(goals * 3.8 + np.random.normal(25, 10) * play_ratio, goals, 90))
                shots_on_target = int(np.clip(shots * np.random.uniform(0.35, 0.48), goals, shots))
                passes = int(np.clip(np.random.normal(1650 * play_ratio, 350), 300, 2800))
                pass_accuracy = round(float(np.clip(np.random.normal(84 + 6 * quality_factor, 3.5), 70.0, 94.5)), 1)
                key_passes = int(np.clip(assists * 2.8 + np.random.normal(28, 12) * play_ratio, assists, 110))
                chances_created = int(key_passes + np.random.normal(12, 6) * play_ratio)
                
                tackles = int(np.clip(np.random.normal(65 * play_ratio if d_pos == "CDM" else 42 * play_ratio, 15), 8, 120))
                interceptions = int(np.clip(np.random.normal(45 * play_ratio if d_pos == "CDM" else 30 * play_ratio, 10), 5, 85))
                clearances = int(np.clip(np.random.normal(25 * play_ratio, 8), 2, 65))
                blocks = int(np.clip(np.random.normal(12 * play_ratio, 5), 1, 35))
                recoveries = int(np.clip(np.random.normal(160 * play_ratio, 35), 30, 280))
                aerial_duels_won = int(np.clip(np.random.normal(32 * play_ratio, 12), 4, 80))
                
                dribbles = int(np.clip(np.random.normal(55 * quality_factor * play_ratio, 18), 8, 130))
                successful_dribbles = int(np.clip(dribbles * np.random.uniform(0.55, 0.72), 4, dribbles))
                progressive_carries = int(np.clip(np.random.normal(75 * quality_factor * play_ratio, 22), 12, 160))
                progressive_passes = int(np.clip(np.random.normal(140 * quality_factor * play_ratio, 40), 25, 290))
                ball_losses = int(np.clip(np.random.normal(75 * play_ratio, 18), 15, 140))
                
                saves, save_percentage, clean_sheets, goals_conceded, penalty_saves = 0, 0.0, 0, 0, 0
                
            elif pos == "DF":
                goals = int(np.clip(np.random.normal(2 * quality_factor * play_ratio, 1.5), 0, 8))
                assists = int(np.clip(np.random.normal(2 * quality_factor * play_ratio if d_pos == "CB" else 6 * quality_factor * play_ratio, 2), 0, 14))
                shots = int(np.clip(goals * 3.0 + np.random.normal(10, 5) * play_ratio, goals, 45))
                shots_on_target = int(np.clip(shots * np.random.uniform(0.28, 0.42), goals, shots))
                passes = int(np.clip(np.random.normal(1450 * play_ratio, 300), 250, 2600))
                pass_accuracy = round(float(np.clip(np.random.normal(83 + 5 * quality_factor, 3.5), 70.0, 93.0)), 1)
                key_passes = int(np.clip(assists * 2.0 + np.random.normal(10, 6) * play_ratio, assists, 55))
                chances_created = int(key_passes + np.random.normal(5, 3) * play_ratio)
                
                tackles = int(np.clip(np.random.normal(58 * play_ratio, 14), 10, 115))
                interceptions = int(np.clip(np.random.normal(48 * play_ratio, 12), 8, 95))
                clearances = int(np.clip(np.random.normal(110 * play_ratio if d_pos == "CB" else 45 * play_ratio, 25), 15, 210))
                blocks = int(np.clip(np.random.normal(26 * play_ratio if d_pos == "CB" else 10 * play_ratio, 7), 2, 55))
                recoveries = int(np.clip(np.random.normal(140 * play_ratio, 30), 25, 240))
                aerial_duels_won = int(np.clip(np.random.normal(68 * play_ratio if d_pos == "CB" else 22 * play_ratio, 18), 5, 140))
                
                dribbles = int(np.clip(np.random.normal(30 * quality_factor * play_ratio if d_pos != "CB" else 12 * quality_factor * play_ratio, 10), 2, 80))
                successful_dribbles = int(np.clip(dribbles * np.random.uniform(0.58, 0.75), 1, dribbles))
                progressive_carries = int(np.clip(np.random.normal(50 * play_ratio if d_pos != "CB" else 25 * play_ratio, 15), 5, 120))
                progressive_passes = int(np.clip(np.random.normal(90 * quality_factor * play_ratio, 25), 15, 200))
                ball_losses = int(np.clip(np.random.normal(45 * play_ratio, 12), 8, 90))
                
                saves, save_percentage, clean_sheets, goals_conceded, penalty_saves = 0, 0.0, 0, 0, 0
                
            else:  # GK
                goals, assists, shots, shots_on_target, key_passes, chances_created = 0, 0, 0, 0, 0, 0
                passes = int(np.clip(np.random.normal(850 * play_ratio, 180), 150, 1400))
                pass_accuracy = round(float(np.clip(np.random.normal(70 + 7 * quality_factor, 5), 50.0, 88.0)), 1)
                tackles, interceptions, blocks = 1, 2, 0
                clearances = int(np.clip(np.random.normal(25 * play_ratio, 8), 4, 55))
                recoveries = int(np.clip(np.random.normal(110 * play_ratio, 25), 20, 180))
                aerial_duels_won = int(np.clip(np.random.normal(18 * play_ratio, 6), 2, 40))
                dribbles, successful_dribbles, progressive_carries = 2, 2, 4
                progressive_passes = int(np.clip(np.random.normal(60 * quality_factor * play_ratio, 18), 10, 120))
                ball_losses = int(np.clip(np.random.normal(30 * play_ratio, 8), 5, 60))
                
                saves = int(np.clip(np.random.normal(95 * play_ratio, 20), 15, 160))
                save_percentage = round(float(np.clip(np.random.normal(68 + 8 * quality_factor, 4.0), 55.0, 83.0)), 1)
                clean_sheets = int(np.clip(np.random.normal(11 * quality_factor * play_ratio, 3.5), 1, 22))
                goals_conceded = int(np.clip(np.random.normal(38 * (1.3 - 0.5 * quality_factor) * play_ratio, 8), 12, 65))
                penalty_saves = int(np.clip(np.random.poisson(1.2), 0, 5))
                
            defensive_actions = tackles + interceptions + clearances + blocks
            
            # Physical and discipline
            distance_covered = round(float(np.clip(minutes_played * np.random.uniform(0.105, 0.122), 12.0, 420.0)), 1)
            sprint_count = int(np.clip(minutes_played * np.random.uniform(0.18, 0.35), 20, 1100))
            fouls = int(np.clip(np.random.normal(28 * play_ratio, 8), 2, 65))
            yellow_cards = int(np.clip(fouls * np.random.uniform(0.10, 0.18), 0, 14))
            red_cards = int(1 if random.random() < 0.08 * (yellow_cards / 6.0) else 0)
            
            # Market value calculation in Millions Euro (€M)
            # Age modifier: peaks at 23-27
            age_mult = 1.0
            if season_age <= 21:
                age_mult = 1.45
            elif season_age <= 25:
                age_mult = 1.35
            elif season_age <= 28:
                age_mult = 1.10
            elif season_age <= 31:
                age_mult = 0.65
            else:
                age_mult = 0.30
                
            raw_mv = np.exp((season_rating - 60) / 6.8) * age_mult * (1.1 if pos in ["FW", "MF"] else 0.85)
            market_value = round(float(np.clip(raw_mv + np.random.normal(0, raw_mv * 0.15), 0.5, 185.0)), 1)
            
            # Future Performance Rating (target for next season)
            # Calculated with age growth curve, current rating momentum, minutes played, and variance
            if season_age < 23:
                next_delta = np.random.normal(2.2, 1.0)
            elif season_age <= 27:
                next_delta = np.random.normal(0.6, 0.8)
            elif season_age <= 30:
                next_delta = np.random.normal(-0.5, 0.8)
            elif season_age <= 33:
                next_delta = np.random.normal(-1.6, 1.0)
            else:
                next_delta = np.random.normal(-2.8, 1.2)
                
            # If player had very high minutes & high performance, they get a boost
            perf_boost = (season_rating - 75) * 0.1
            minutes_penalty = -1.2 if minutes_played < 800 else 0.0
            injury_penalty = -0.015 * injury_days
            
            future_rating = round(float(np.clip(season_rating + next_delta + perf_boost + minutes_penalty + injury_penalty, 50.0, 95.0)), 1)
            
            record = {
                "player_id": profile["player_id"],
                "player_name": profile["player_name"],
                "age": season_age,
                "nationality": profile["nationality"],
                "position": pos,
                "detailed_position": d_pos,
                "club": profile["club"],
                "league": profile["league"],
                "season": season,
                
                # Playing Stats
                "appearances": appearances,
                "minutes_played": minutes_played,
                "starts": starts,
                "goals": goals,
                "assists": assists,
                "shots": shots,
                "shots_on_target": shots_on_target,
                "passes": passes,
                "pass_accuracy": pass_accuracy,
                "key_passes": key_passes,
                "chances_created": chances_created,
                
                # Defensive Stats
                "tackles": tackles,
                "interceptions": interceptions,
                "clearances": clearances,
                "blocks": blocks,
                "recoveries": recoveries,
                "aerial_duels_won": aerial_duels_won,
                "defensive_actions": defensive_actions,
                
                # Possession / Progression
                "dribbles": dribbles,
                "successful_dribbles": successful_dribbles,
                "progressive_carries": progressive_carries,
                "progressive_passes": progressive_passes,
                "ball_losses": ball_losses,
                
                # Physical / Discipline
                "distance_covered": distance_covered,
                "sprint_count": sprint_count,
                "fouls": fouls,
                "yellow_cards": yellow_cards,
                "red_cards": red_cards,
                
                # Goalkeeping
                "saves": saves,
                "save_percentage": save_percentage,
                "clean_sheets": clean_sheets,
                "goals_conceded": goals_conceded,
                "penalty_saves": penalty_saves,
                
                # Performance & Valuation
                "player_rating": season_rating,
                "market_value": market_value,
                "injury_days": injury_days,
                "previous_season_rating": round(float(prev_rating), 1),
                "future_performance_rating": future_rating
            }
            
            records.append(record)
            prev_rating = season_rating
            
    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    raw_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    out_path = os.path.join(raw_dir, "players.csv")
    
    print("Generating comprehensive football dataset...")
    df = generate_player_database(n_synthetic_players=580)
    df.to_csv(out_path, index=False)
    print(f"Successfully generated {len(df)} player-season records across {df['player_id'].nunique()} unique players.")
    print(f"Saved to: {out_path}")
