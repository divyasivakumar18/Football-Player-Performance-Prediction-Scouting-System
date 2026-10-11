"""
SQLite Relational Database Manager for Football Scouting System.
Handles schema initialization, data population, and parameterized SQL queries.
"""

import os
import sqlite3
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DB_DIR = os.path.join(BASE_DIR, "database")
DB_PATH = os.path.join(DB_DIR, "football_scouting.db")


class FootballDatabase:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_database()
        
    def get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)
        
    def init_database(self):
        """Initializes database tables."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # 1. Clubs Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS clubs (
            club_id INTEGER PRIMARY KEY AUTOINCREMENT,
            club_name TEXT UNIQUE NOT NULL,
            league TEXT NOT NULL
        );
        """)
        
        # 2. Players Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS players (
            player_id TEXT PRIMARY KEY,
            player_name TEXT NOT NULL,
            nationality TEXT,
            position TEXT,
            detailed_position TEXT,
            club TEXT,
            league TEXT,
            market_value REAL,
            FOREIGN KEY (club) REFERENCES clubs(club_name)
        );
        """)
        
        # 3. Performance Stats Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS performance_stats (
            stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id TEXT NOT NULL,
            season TEXT NOT NULL,
            age INTEGER,
            appearances INTEGER,
            minutes_played INTEGER,
            goals INTEGER,
            assists INTEGER,
            shots INTEGER,
            shots_on_target INTEGER,
            passes INTEGER,
            pass_accuracy REAL,
            key_passes INTEGER,
            chances_created INTEGER,
            tackles INTEGER,
            interceptions INTEGER,
            clearances INTEGER,
            recoveries INTEGER,
            dribbles INTEGER,
            successful_dribbles INTEGER,
            progressive_passes INTEGER,
            player_rating REAL,
            future_performance_rating REAL,
            attacking_score REAL,
            defensive_score REAL,
            passing_score REAL,
            progression_score REAL,
            overall_performance_score REAL,
            potential_score REAL,
            potential_tier TEXT,
            FOREIGN KEY (player_id) REFERENCES players(player_id)
        );
        """)
        
        # 4. Predictions Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id TEXT,
            player_name TEXT NOT NULL,
            age INTEGER,
            position TEXT,
            predicted_rating REAL,
            confidence_lower REAL,
            confidence_upper REAL,
            potential_score REAL,
            potential_tier TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        # 5. Scouting Shortlists Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS scouting_results (
            result_id INTEGER PRIMARY KEY AUTOINCREMENT,
            search_label TEXT,
            player_id TEXT,
            player_name TEXT,
            position TEXT,
            club TEXT,
            match_score REAL,
            rating REAL,
            potential REAL,
            market_value REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        conn.commit()
        conn.close()
        
    def populate_from_dataframe(self, df: pd.DataFrame):
        """Populates clubs, players, and performance_stats tables from processed dataframe."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # 1. Populate Clubs
        clubs_data = df[["club", "league"]].drop_duplicates()
        for _, row in clubs_data.iterrows():
            cursor.execute("""
            INSERT OR IGNORE INTO clubs (club_name, league)
            VALUES (?, ?)
            """, (row["club"], row["league"]))
            
        # 2. Populate Players (latest record)
        latest_players = df.sort_values("season").groupby("player_id").last().reset_index()
        for _, row in latest_players.iterrows():
            cursor.execute("""
            INSERT OR REPLACE INTO players (player_id, player_name, nationality, position, detailed_position, club, league, market_value)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row["player_id"], row["player_name"], row["nationality"],
                row["position"], row.get("detailed_position", row["position"]),
                row["club"], row["league"], row["market_value"]
            ))
            
        # 3. Populate Performance Stats
        for _, row in df.iterrows():
            cursor.execute("""
            INSERT INTO performance_stats (
                player_id, season, age, appearances, minutes_played, goals, assists,
                shots, shots_on_target, passes, pass_accuracy, key_passes, chances_created,
                tackles, interceptions, clearances, recoveries, dribbles, successful_dribbles,
                progressive_passes, player_rating, future_performance_rating, attacking_score,
                defensive_score, passing_score, progression_score, overall_performance_score,
                potential_score, potential_tier
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row["player_id"], row["season"], int(row["age"]), int(row["appearances"]), int(row["minutes_played"]),
                int(row["goals"]), int(row["assists"]), int(row["shots"]), int(row["shots_on_target"]), int(row["passes"]),
                float(row["pass_accuracy"]), int(row["key_passes"]), int(row["chances_created"]),
                int(row["tackles"]), int(row["interceptions"]), int(row["clearances"]), int(row["recoveries"]),
                int(row["dribbles"]), int(row["successful_dribbles"]), int(row["progressive_passes"]),
                float(row["player_rating"]), float(row.get("future_performance_rating", row["player_rating"])),
                float(row.get("attacking_score", 50.0)), float(row.get("defensive_score", 50.0)),
                float(row.get("passing_score", 50.0)), float(row.get("progression_score", 50.0)),
                float(row.get("overall_performance_score", 50.0)), float(row.get("potential_score", 50.0)),
                str(row.get("potential_tier", "Promising"))
            ))
            
        conn.commit()
        conn.close()
        print(f"Successfully loaded {len(df)} performance records into SQLite database.")

    def run_query(self, query: str, params: tuple = ()) -> pd.DataFrame:
        """Executes a SQL query and returns results as a DataFrame."""
        conn = self.get_connection()
        try:
            df = pd.read_sql_query(query, conn, params=params)
            return df
        finally:
            conn.close()

    def get_top_rated_players(self, season: str = "2024-2025", limit: int = 15) -> pd.DataFrame:
        query = """
        SELECT p.player_name, p.position, p.club, p.league, s.player_rating, s.goals, s.assists, s.potential_score, s.potential_tier
        FROM performance_stats s
        JOIN players p ON s.player_id = p.player_id
        WHERE s.season = ?
        ORDER BY s.player_rating DESC
        LIMIT ?;
        """
        return self.run_query(query, (season, limit))

    def get_top_young_prospects(self, season: str = "2024-2025", max_age: int = 21, limit: int = 15) -> pd.DataFrame:
        query = """
        SELECT p.player_name, s.age, p.position, p.club, p.league, s.potential_score, s.potential_tier, s.player_rating, p.market_value
        FROM performance_stats s
        JOIN players p ON s.player_id = p.player_id
        WHERE s.season = ? AND s.age <= ?
        ORDER BY s.potential_score DESC
        LIMIT ?;
        """
        return self.run_query(query, (season, max_age, limit))

    def get_top_scorers(self, season: str = "2024-2025", limit: int = 15) -> pd.DataFrame:
        query = """
        SELECT p.player_name, p.club, p.league, s.goals, s.assists, s.shots_on_target, s.attacking_score, s.player_rating
        FROM performance_stats s
        JOIN players p ON s.player_id = p.player_id
        WHERE s.season = ?
        ORDER BY s.goals DESC, s.assists DESC
        LIMIT ?;
        """
        return self.run_query(query, (season, limit))

    def get_top_playmakers(self, season: str = "2024-2025", limit: int = 15) -> pd.DataFrame:
        query = """
        SELECT p.player_name, p.position, p.club, p.league, s.assists, s.key_passes, s.chances_created, s.pass_accuracy, s.passing_score
        FROM performance_stats s
        JOIN players p ON s.player_id = p.player_id
        WHERE s.season = ?
        ORDER BY s.passing_score DESC, s.assists DESC
        LIMIT ?;
        """
        return self.run_query(query, (season, limit))

    def get_top_defenders(self, season: str = "2024-2025", limit: int = 15) -> pd.DataFrame:
        query = """
        SELECT p.player_name, p.position, p.club, p.league, s.tackles, s.interceptions, s.clearances, s.defensive_score, s.player_rating
        FROM performance_stats s
        JOIN players p ON s.player_id = p.player_id
        WHERE s.season = ? AND p.position = 'DF'
        ORDER BY s.defensive_score DESC
        LIMIT ?;
        """
        return self.run_query(query, (season, limit))


if __name__ == "__main__":
    db = FootballDatabase()
    proc_csv = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "players_processed.csv")
    if os.path.exists(proc_csv):
        df_p = pd.read_csv(proc_csv)
        db.populate_from_dataframe(df_p)
        print("Sample Top Prospects from SQL:")
        print(db.get_top_young_prospects(limit=5))
