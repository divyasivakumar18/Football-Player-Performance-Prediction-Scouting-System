"""
FOOTBALL IQ - Player Performance Prediction & Scouting Intelligence Platform
Interactive Streamlit Dashboard
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.database import FootballDatabase
from src.prediction import PerformancePredictionEngine
from src.scouting import (
    recommend_scouted_players,
    recommend_alternative_position,
    compute_strength_weaknesses,
    find_hidden_gems
)

# Page configuration
st.set_page_config(
    page_title="FOOTBALL IQ | Scouting & Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
<style>
    /* Dark sleek theme accents */
    .main-header {
        font-family: 'Segoe UI', sans-serif;
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00FF87, #60EFFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #A0AEC0;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #1A202C;
        border: 1px solid #2D3748;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #718096;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #00FF87;
        margin-top: 0.3rem;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #CBD5E0;
    }
    .tier-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .tier-elite { background-color: #7928CA; color: #FFF; }
    .tier-high { background-color: #0070F3; color: #FFF; }
    .tier-promising { background-color: #00DF72; color: #000; }
    .tier-dev { background-color: #F5A623; color: #000; }
    .tier-low { background-color: #4A5568; color: #FFF; }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_all_data():
    """Loads processed dataset and database connection."""
    proc_csv = os.path.join(BASE_DIR, "data", "processed", "players_processed.csv")
    if not os.path.exists(proc_csv):
        # Fallback to raw if processed is missing
        from src.data_processing import load_and_preprocess
        raw_csv = os.path.join(BASE_DIR, "data", "raw", "players.csv")
        df, _ = load_and_preprocess(raw_csv)
    else:
        df = pd.read_csv(proc_csv)
        
    db = FootballDatabase()
    return df, db


@st.cache_resource
def load_ml_engine():
    """Loads ML prediction engine and model metadata."""
    return PerformancePredictionEngine()


# Load global data and engine
df_players, db = load_all_data()
ml_engine = load_ml_engine()
latest_season = "2024-2025"
df_current = df_players[df_players["season"] == latest_season].copy()
if len(df_current) == 0:
    df_current = df_players.copy()


# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/53/53283.png", width=60)
    st.markdown("<h2 style='margin-bottom:0;'>FOOTBALL IQ</h2>", unsafe_allow_html=True)
    st.caption("AI Performance Prediction & Scouting")
    st.markdown("---")
    
    nav_option = st.radio(
        "Navigation Menu",
        [
            "🏠 Dashboard",
            "🔍 Player Explorer",
            "👤 Player Profile",
            "⚔️ Player Comparison",
            "🔮 AI Performance Prediction",
            "🎯 Scouting & Shortlist",
            "🧬 Player Archetypes & Clustering",
            "📊 Model Performance & XAI",
            "🗄️ SQL Scouting Console",
            "ℹ️ About Platform"
        ]
    )
    
    st.markdown("---")
    st.markdown("### Quick Global Scope")
    st.info(f"**Database:** {len(df_players):,} Total Records\n\n**Active Season:** {latest_season}\n\n**Unique Players:** {df_players['player_id'].nunique()}")


# -------------------------------------------------------------
# 1. 🏠 DASHBOARD (HOME)
# -------------------------------------------------------------
if nav_option == "🏠 Dashboard":
    st.markdown("<div class='main-header'>FOOTBALL IQ DASHBOARD</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Executive overview of top prospects, positional leaders, and squad performance across European leagues.</div>", unsafe_allow_html=True)
    
    # KPI Cards
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    
    top_prospect_row = df_current[df_current["age"] <= 21].sort_values("potential_score", ascending=False).iloc[0]
    highest_pred_row = df_current.sort_values("future_performance_rating", ascending=False).iloc[0]
    
    with kpi1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>Total Players</div>
            <div class='metric-val'>{df_players['player_id'].nunique()}</div>
            <div class='metric-sub'>Across 5 Top Leagues</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>Clubs Analyzed</div>
            <div class='metric-val'>{df_players['club'].nunique()}</div>
            <div class='metric-sub'>Elite Football Clubs</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>Average Rating</div>
            <div class='metric-val'>{df_current['player_rating'].mean():.1f}</div>
            <div class='metric-sub'>League Benchmark</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>Top Prospect</div>
            <div class='metric-val' style='font-size:1.2rem; color:#60EFFF;'>{top_prospect_row['player_name']}</div>
            <div class='metric-sub'>Pot: <b>{top_prospect_row['potential_score']:.1f}</b> ({top_prospect_row['club']})</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi5:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>Highest Predicted</div>
            <div class='metric-val' style='font-size:1.2rem; color:#FFDF00;'>{highest_pred_row['player_name']}</div>
            <div class='metric-sub'>Predicted: <b>{highest_pred_row['future_performance_rating']:.1f}</b></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Leaderboards Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌟 Top 10 Young Prospects",
        "🎯 Top Attackers",
        "🛡️ Top Defenders",
        "🪄 Top Playmakers",
        "💎 Hidden Gems"
    ])
    
    with tab1:
        top_prospects = df_current[df_current["age"] <= 21].sort_values("potential_score", ascending=False).head(10)
        fig_pot = px.bar(
            top_prospects,
            x="player_name",
            y="potential_score",
            color="position",
            hover_data=["club", "age", "player_rating", "market_value"],
            labels={"player_name": "Player", "potential_score": "Potential Score", "position": "Role"},
            title="Top 10 Under-21 Elite Prospects (by AI Potential Score)",
            text="potential_score",
            color_discrete_map={"FW": "#FF4B4B", "MF": "#00FF87", "DF": "#0070F3", "GK": "#FFD700"}
        )
        fig_pot.update_layout(height=400, xaxis_tickangle=-30)
        st.plotly_chart(fig_pot, use_container_width=True)
        st.dataframe(
            top_prospects[["player_name", "age", "position", "club", "league", "player_rating", "future_performance_rating", "potential_score", "potential_tier", "market_value"]],
            use_container_width=True
        )

    with tab2:
        top_atk = df_current[df_current["position"] == "FW"].sort_values("goals", ascending=False).head(10)
        fig_atk = px.scatter(
            top_atk,
            x="goals",
            y="assists",
            size="attacking_score",
            color="club",
            text="player_name",
            title="Top Attackers: Goals vs Assists (Bubble Size = Attacking Score)",
            labels={"goals": "Season Goals", "assists": "Season Assists"}
        )
        fig_atk.update_traces(textposition="top center")
        fig_atk.update_layout(height=400)
        st.plotly_chart(fig_atk, use_container_width=True)
        st.dataframe(top_atk[["player_name", "club", "goals", "assists", "shots_on_target", "attacking_score", "player_rating"]], use_container_width=True)

    with tab3:
        top_def = df_current[df_current["position"] == "DF"].sort_values("defensive_score", ascending=False).head(10)
        fig_def = px.bar(
            top_def,
            x="player_name",
            y="defensive_score",
            color="club",
            title="Top Defensive Anchors (Ranked by Defensive Composite Score)",
            text="defensive_score"
        )
        fig_def.update_layout(height=400, xaxis_tickangle=-30)
        st.plotly_chart(fig_def, use_container_width=True)
        st.dataframe(top_def[["player_name", "club", "tackles", "interceptions", "clearances", "defensive_score", "player_rating"]], use_container_width=True)

    with tab4:
        top_mf = df_current[df_current["position"] == "MF"].sort_values("passing_score", ascending=False).head(10)
        fig_mf = px.scatter(
            top_mf,
            x="pass_accuracy",
            y="key_passes",
            size="passing_score",
            color="club",
            text="player_name",
            title="Midfield Architects: Pass Accuracy vs Key Passes",
            labels={"pass_accuracy": "Pass Accuracy (%)", "key_passes": "Key Passes"}
        )
        fig_mf.update_traces(textposition="top center")
        fig_mf.update_layout(height=400)
        st.plotly_chart(fig_mf, use_container_width=True)
        st.dataframe(top_mf[["player_name", "club", "pass_accuracy", "key_passes", "progressive_passes", "passing_score", "player_rating"]], use_container_width=True)

    with tab5:
        gems_df = find_hidden_gems(df_current)
        st.info("💡 **Hidden Gems** are high-potential young players (Age ≤ 23) with market values ≤ €35M who outperform their price bracket.")
        st.dataframe(
            gems_df[["player_name", "age", "position", "club", "league", "market_value", "potential_score", "future_performance_rating", "gem_value_ratio"]].head(15),
            use_container_width=True
        )


# -------------------------------------------------------------
# 2. 🔍 PLAYER EXPLORER & SEARCH
# -------------------------------------------------------------
elif nav_option == "🔍 Player Explorer":
    st.markdown("<div class='main-header'>PLAYER DATABASE EXPLORER</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Filter, search, and export multi-season player performance and tactical records.</div>", unsafe_allow_html=True)
    
    # Filter Row 1
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sel_league = st.selectbox("Filter League", ["All"] + sorted(df_players["league"].unique().tolist()))
    with c2:
        if sel_league != "All":
            clubs_in_league = sorted(df_players[df_players["league"] == sel_league]["club"].unique().tolist())
        else:
            clubs_in_league = sorted(df_players["club"].unique().tolist())
        sel_club = st.selectbox("Filter Club", ["All"] + clubs_in_league)
    with c3:
        sel_pos = st.selectbox("Filter Position", ["All", "FW", "MF", "DF", "GK"])
    with c4:
        sel_season = st.selectbox("Filter Season", ["All"] + sorted(df_players["season"].unique().tolist(), reverse=True))

    # Filter Row 2
    c5, c6, c7 = st.columns(3)
    with c5:
        age_range = st.slider("Age Range", int(df_players["age"].min()), int(df_players["age"].max()), (16, 36))
    with c6:
        rating_range = st.slider("Rating Range", float(df_players["player_rating"].min()), float(df_players["player_rating"].max()), (60.0, 95.0))
    with c7:
        search_query = st.text_input("🔍 Search Player by Name", "")

    # Apply Filters
    filtered_df = df_players.copy()
    if sel_league != "All":
        filtered_df = filtered_df[filtered_df["league"] == sel_league]
    if sel_club != "All":
        filtered_df = filtered_df[filtered_df["club"] == sel_club]
    if sel_pos != "All":
        filtered_df = filtered_df[filtered_df["position"] == sel_pos]
    if sel_season != "All":
        filtered_df = filtered_df[filtered_df["season"] == sel_season]
    filtered_df = filtered_df[
        (filtered_df["age"] >= age_range[0]) & (filtered_df["age"] <= age_range[1]) &
        (filtered_df["player_rating"] >= rating_range[0]) & (filtered_df["player_rating"] <= rating_range[1])
    ]
    if search_query.strip():
        filtered_df = filtered_df[filtered_df["player_name"].str.contains(search_query.strip(), case=False, na=False)]

    st.markdown(f"**Found {len(filtered_df):,} matching player-season records**")
    
    display_cols = [
        "player_name", "age", "position", "detailed_position", "club", "league", "season",
        "appearances", "minutes_played", "goals", "assists", "pass_accuracy",
        "player_rating", "future_performance_rating", "potential_score", "potential_tier",
        "market_value", "attacking_score", "defensive_score", "passing_score", "progression_score"
    ]
    avail_cols = [c for c in display_cols if c in filtered_df.columns]
    
    st.dataframe(filtered_df[avail_cols], use_container_width=True)
    
    csv_data = filtered_df[avail_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Filtered Data (CSV)",
        data=csv_data,
        file_name="football_iq_scouting_export.csv",
        mime="text/csv"
    )


# -------------------------------------------------------------
# 3. 👤 PLAYER PROFILE & SCOUT CARD
# -------------------------------------------------------------
elif nav_option == "👤 Player Profile":
    st.markdown("<div class='main-header'>SCOUTING DOSSIER & PLAYER PROFILE</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Deep tactical breakdown, percentile benchmark against position peers, and potential trajectory.</div>", unsafe_allow_html=True)
    
    all_player_names = sorted(df_current["player_name"].unique().tolist())
    default_name = "Erling Haaland" if "Erling Haaland" in all_player_names else all_player_names[0]
    
    sel_player_name = st.selectbox("Select Player to Inspect", all_player_names, index=all_player_names.index(default_name) if default_name in all_player_names else 0)
    
    player_history = df_players[df_players["player_name"] == sel_player_name].sort_values("season")
    curr_p = player_history.iloc[-1]
    
    # Top Profile Header Card
    h1, h2, h3, h4, h5 = st.columns(5)
    with h1:
        st.markdown(f"### {curr_p['player_name']}")
        st.caption(f"**Club:** {curr_p['club']} | **League:** {curr_p['league']}")
        st.caption(f"**Nationality:** {curr_p['nationality']} | **Age:** {curr_p['age']}")
    with h2:
        st.metric("Current Rating", f"{curr_p['player_rating']:.1f}", f"{curr_p['player_rating'] - curr_p['previous_season_rating']:+.1f} YoY")
    with h3:
        st.metric("Predicted Next Rating", f"{curr_p['future_performance_rating']:.1f}", f"{curr_p['future_performance_rating'] - curr_p['player_rating']:+.1f} Pred")
    with h4:
        st.metric("Potential Score", f"{curr_p['potential_score']:.1f}", curr_p['potential_tier'])
    with h5:
        st.metric("Market Value", f"€{curr_p['market_value']:.1f}M", f"{curr_p['detailed_position']} ({curr_p['position']})")

    st.markdown("---")
    
    # Detailed Analysis Columns
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("#### 🕸️ Tactical Radar Profile")
        radar_categories = ["Attacking", "Defensive", "Passing", "Progression", "Overall Rating", "Potential"]
        radar_values = [
            curr_p.get("attacking_score", 50.0),
            curr_p.get("defensive_score", 50.0),
            curr_p.get("passing_score", 50.0),
            curr_p.get("progression_score", 50.0),
            curr_p.get("overall_performance_score", 50.0),
            curr_p.get("potential_score", 50.0)
        ]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_values + [radar_values[0]],
            theta=radar_categories + [radar_categories[0]],
            fill='toself',
            fillcolor='rgba(0, 255, 135, 0.25)',
            line=dict(color='#00FF87', width=2),
            name=curr_p['player_name']
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10)),
            ),
            showlegend=False,
            height=360,
            margin=dict(l=40, r=40, t=30, b=30)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_right:
        st.markdown("#### 📈 Multi-Season Rating Trajectory")
        fig_trend = px.line(
            player_history,
            x="season",
            y="player_rating",
            markers=True,
            title="Career Performance Evolution",
            labels={"season": "Season", "player_rating": "Performance Rating"}
        )
        # Add future projection point
        next_season_label = "2025-2026 (Pred)"
        fig_trend.add_trace(go.Scatter(
            x=[player_history["season"].iloc[-1], next_season_label],
            y=[player_history["player_rating"].iloc[-1], curr_p["future_performance_rating"]],
            mode="lines+markers",
            line=dict(color="#FFDF00", dash="dash", width=2),
            name="AI Future Prediction"
        ))
        fig_trend.update_layout(height=360)
        st.plotly_chart(fig_trend, use_container_width=True)

    # Strengths & Weaknesses (Benchmarked against exact position peers)
    st.markdown("#### 🎯 Position-Specific Strength & Weakness Benchmarking")
    pos_peers = df_current[df_current["position"] == curr_p["position"]]
    sw_data = compute_strength_weaknesses(curr_p, pos_peers)
    
    col_str, col_weak, col_alt = st.columns(3)
    with col_str:
        st.markdown("##### 🟢 Key Strengths (≥ 70th Percentile)")
        for s in sw_data["strengths"]:
            st.success(f"✓ {s}")
    with col_weak:
        st.markdown("##### 🔴 Weaknesses (≤ 30th Percentile)")
        for w in sw_data["weaknesses"]:
            st.error(f"✗ {w}")
    with col_alt:
        alt_data = recommend_alternative_position(curr_p)
        st.markdown("##### 🔄 Tactical Versatility & Alternate Roles")
        st.info(f"**Recommended Role:** {alt_data['recommended_alternative']}\n\n**Tactical Reason:** {alt_data['tactical_reason']}")


# -------------------------------------------------------------
# 4. ⚔️ PLAYER COMPARISON
# -------------------------------------------------------------
elif nav_option == "⚔️ Player Comparison":
    st.markdown("<div class='main-header'>HEAD-TO-HEAD PLAYER COMPARISON</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Select 2 to 4 players for multi-dimensional tactical radar and statistical comparison.</div>", unsafe_allow_html=True)
    
    all_players = sorted(df_current["player_name"].unique().tolist())
    default_selection = ["Erling Haaland", "Kylian Mbappé"] if ("Erling Haaland" in all_players and "Kylian Mbappé" in all_players) else all_players[:2]
    
    selected_players = st.multiselect("Select Players to Compare (2 - 4)", all_players, default=default_selection)
    
    if len(selected_players) >= 2:
        comp_df = df_current[df_current["player_name"].isin(selected_players)]
        
        # Multi-Radar Chart
        radar_metrics = ["attacking_score", "defensive_score", "passing_score", "progression_score", "overall_performance_score", "potential_score"]
        radar_labels = ["Attacking", "Defensive", "Passing", "Progression", "Overall Rating", "Potential"]
        
        fig_comp_radar = go.Figure()
        colors = ["#00FF87", "#60EFFF", "#FF4B4B", "#FFD700"]
        
        for idx, (_, p_row) in enumerate(comp_df.iterrows()):
            vals = [p_row.get(m, 50.0) for m in radar_metrics]
            fig_comp_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=radar_labels + [radar_labels[0]],
                fill='toself',
                fillcolor=f"rgba{tuple(list(int(colors[idx % len(colors)].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.15])}",
                line=dict(color=colors[idx % len(colors)], width=2),
                name=f"{p_row['player_name']} ({p_row['club']})"
            ))
            
        fig_comp_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            height=450,
            title="Tactical Radar Overlay"
        )
        st.plotly_chart(fig_comp_radar, use_container_width=True)
        
        # Direct Stat Comparison Table
        st.markdown("#### 📊 Head-to-Head Statistics Breakdown")
        stat_compare_cols = [
            "player_name", "position", "club", "age", "player_rating", "future_performance_rating",
            "potential_score", "goals", "assists", "pass_accuracy", "key_passes",
            "tackles", "interceptions", "dribbles", "market_value"
        ]
        st.dataframe(comp_df[stat_compare_cols].set_index("player_name").T, use_container_width=True)
    else:
        st.warning("Please select at least 2 players from the dropdown above to view the comparison.")


# -------------------------------------------------------------
# 5. 🔮 AI PERFORMANCE PREDICTION INTERFACE
# -------------------------------------------------------------
elif nav_option == "🔮 AI Performance Prediction":
    st.markdown("<div class='main-header'>AI PLAYER PERFORMANCE & POTENTIAL PREDICTOR</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Input custom player performance statistics to generate future ratings, 95% confidence intervals, and Explainable AI (XAI) feature attribution.</div>", unsafe_allow_html=True)
    
    with st.form("prediction_form"):
        st.markdown("#### 📝 Player Core Attributes & Match Statistics")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            in_name = st.text_input("Player Name", "Marcus Sterling")
            in_age = st.number_input("Age", min_value=15, max_value=42, value=21)
            in_pos = st.selectbox("Primary Position", ["FW", "MF", "DF", "GK"], index=1)
        with c2:
            in_prev_rating = st.number_input("Previous Season Rating", min_value=50.0, max_value=95.0, value=76.0, step=0.5)
            in_minutes = st.number_input("Minutes Played", min_value=100, max_value=3800, value=2300, step=100)
            in_appearances = st.number_input("Appearances", min_value=2, max_value=38, value=28)
        with c3:
            in_goals = st.number_input("Goals", min_value=0, max_value=50, value=7)
            in_assists = st.number_input("Assists", min_value=0, max_value=30, value=8)
            in_shots = st.number_input("Shots on Target", min_value=0, max_value=80, value=22)
        with c4:
            in_pass_acc = st.number_input("Pass Accuracy (%)", min_value=40.0, max_value=98.0, value=84.5, step=0.5)
            in_key_passes = st.number_input("Key Passes", min_value=0, max_value=120, value=45)
            in_tackles = st.number_input("Tackles & Interceptions", min_value=0, max_value=150, value=52)
            
        submit_pred = st.form_submit_button("⚡ Predict Future Performance & Potential", use_container_width=True)
        
    if submit_pred:
        input_payload = {
            "player_name": in_name,
            "age": in_age,
            "position": in_pos,
            "player_rating": in_prev_rating,
            "previous_season_rating": in_prev_rating,
            "minutes_played": in_minutes,
            "appearances": in_appearances,
            "goals": in_goals,
            "assists": in_assists,
            "shots_on_target": in_shots,
            "pass_accuracy": in_pass_acc,
            "key_passes": in_key_passes,
            "tackles": in_tackles,
            "interceptions": int(in_tackles * 0.7)
        }
        
        pred_res = ml_engine.predict_single(input_payload)
        
        st.markdown("### 🏆 Prediction Results")
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.metric("Predicted Performance Rating", f"{pred_res['predicted_rating']:.1f}", f"±{pred_res['confidence_margin']} Margin")
        with r2:
            st.metric("95% Confidence Interval", f"[{pred_res['confidence_lower']:.1f} — {pred_res['confidence_upper']:.1f}]")
        with r3:
            st.metric("AI Potential Score", f"{pred_res['potential_score']:.1f}")
        with r4:
            st.markdown(f"**Prospect Classification:**")
            tier_class = "tier-elite" if "Elite" in pred_res["potential_tier"] else ("tier-high" if "High" in pred_res["potential_tier"] else "tier-promising")
            st.markdown(f"<span class='tier-badge {tier_class}'>{pred_res['potential_tier']}</span>", unsafe_allow_html=True)
            
        st.markdown("---")
        st.markdown("#### 🔍 Explainable AI (XAI) - Key Factors Influencing This Prediction")
        
        factors_df = pd.DataFrame(pred_res["top_factors"])
        if not factors_df.empty:
            fig_xai = px.bar(
                factors_df,
                x="importance_pct",
                y="display_name",
                orientation="h",
                color="importance_pct",
                title="Model Feature Contribution to Prediction (%)",
                labels={"importance_pct": "Relative Impact (%)", "display_name": "Performance Metric"},
                color_continuous_scale="Viridis"
            )
            fig_xai.update_layout(height=320, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_xai, use_container_width=True)


# -------------------------------------------------------------
# 6. 🎯 SCOUTING & SHORTLIST ENGINE
# -------------------------------------------------------------
elif nav_option == "🎯 Scouting & Shortlist":
    st.markdown("<div class='main-header'>SCOUTING RECOMMENDATION ENGINE</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Configure your squad requirements and budget constraints to compute dynamic Match Scores and recruit top talent.</div>", unsafe_allow_html=True)
    
    with st.expander("🛠️ Configure Scouting Criteria & Club Requirements", expanded=True):
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            sc_pos = st.selectbox("Target Position", ["All", "FW", "MF", "DF", "GK"], index=0)
            sc_min_rating = st.slider("Minimum Current Rating", 55.0, 92.0, 72.0)
        with sc2:
            sc_age = st.slider("Age Requirement", 16, 36, (18, 25))
            sc_budget = st.slider("Maximum Transfer Budget (€M)", 5.0, 180.0, 60.0)
        with sc3:
            sc_goals = st.number_input("Required Min Goals", 0, 30, 3)
            sc_assists = st.number_input("Required Min Assists", 0, 20, 3)
        with sc4:
            sc_pass = st.slider("Min Pass Accuracy (%)", 50.0, 95.0, 75.0)
            sc_def = st.slider("Min Defensive Score", 20.0, 90.0, 35.0)
            
    recs = recommend_scouted_players(
        df_current,
        target_pos=sc_pos,
        min_age=sc_age[0],
        max_age=sc_age[1],
        min_rating=sc_min_rating,
        req_goals=sc_goals,
        req_assists=sc_assists,
        min_pass_acc=sc_pass,
        min_def_score=sc_def,
        max_budget=sc_budget,
        top_n=20
    )
    
    st.markdown(f"### 📋 Ranked Scouting Shortlist ({len(recs)} Recommended Prospects)")
    
    if not recs.empty:
        # Visualizing Match Score
        fig_scout = px.bar(
            recs.head(10),
            x="player_name",
            y="scouting_match_score",
            color="position",
            hover_data=["club", "age", "player_rating", "potential_score", "market_value"],
            text="scouting_match_score",
            title="Top 10 Scouting Matches by Dynamic Match Score (0 - 100)"
        )
        fig_scout.update_layout(height=380, xaxis_tickangle=-30)
        st.plotly_chart(fig_scout, use_container_width=True)
        
        display_scout_cols = [
            "player_name", "scouting_match_score", "position", "detailed_position", "age", "club",
            "player_rating", "potential_score", "potential_tier", "market_value", "goals", "assists", "pass_accuracy"
        ]
        st.dataframe(recs[display_scout_cols], use_container_width=True)
        
        scout_csv = recs[display_scout_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Export Scouting Shortlist (CSV)",
            data=scout_csv,
            file_name="scouting_shortlist.csv",
            mime="text/csv"
        )
    else:
        st.warning("No players matched the strict combination of filters. Try widening the age range or budget.")


# -------------------------------------------------------------
# 7. 🧬 PLAYER ARCHETYPES & CLUSTERING
# -------------------------------------------------------------
elif nav_option == "🧬 Player Archetypes & Clustering":
    st.markdown("<div class='main-header'>TACTICAL ARCHETYPES & PLAYER CLUSTERING</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Unsupervised K-Means clustering and PCA projection identifying 5 tactical player archetypes.</div>", unsafe_allow_html=True)
    
    import joblib
    cluster_model_path = os.path.join(BASE_DIR, "models", "clustering_model.pkl")
    pca_model_path = os.path.join(BASE_DIR, "models", "pca_model.pkl")
    scaler_path = os.path.join(BASE_DIR, "models", "cluster_scaler.pkl")
    
    if os.path.exists(cluster_model_path) and os.path.exists(pca_model_path):
        kmeans = joblib.load(cluster_model_path)
        pca = joblib.load(pca_model_path)
        c_scaler = joblib.load(scaler_path)
        
        from src.model_training import CLUSTERING_FEATURES, ARCHETYPE_NAMES
        feat_cols = [c for c in CLUSTERING_FEATURES if c in df_current.columns]
        X_scaled = c_scaler.transform(df_current[feat_cols].fillna(0))
        cluster_preds = kmeans.predict(X_scaled)
        pca_coords = pca.transform(X_scaled)
        
        df_cluster_plot = df_current.copy()
        df_cluster_plot["cluster_id"] = cluster_preds
        df_cluster_plot["archetype"] = df_cluster_plot["cluster_id"].map(ARCHETYPE_NAMES)
        df_cluster_plot["PCA_1"] = pca_coords[:, 0]
        df_cluster_plot["PCA_2"] = pca_coords[:, 1]
        
        fig_pca = px.scatter(
            df_cluster_plot,
            x="PCA_1",
            y="PCA_2",
            color="archetype",
            hover_name="player_name",
            hover_data=["club", "position", "player_rating", "goals", "assists", "tackles"],
            title="2D PCA Projection of Player Tactical Archetypes (K-Means k=5)",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_pca.update_layout(height=520)
        st.plotly_chart(fig_pca, use_container_width=True)
        
        # Archetype Summary Cards
        st.markdown("#### 📊 Archetype Breakdown")
        a_cols = st.columns(5)
        for idx, (c_id, name) in enumerate(ARCHETYPE_NAMES.items()):
            subset = df_cluster_plot[df_cluster_plot["cluster_id"] == c_id]
            with a_cols[idx]:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-title'>{name}</div>
                    <div class='metric-val' style='font-size:1.5rem;'>{len(subset)}</div>
                    <div class='metric-sub'>Avg Rating: <b>{subset['player_rating'].mean():.1f}</b></div>
                </div>
                """, unsafe_allow_html=True)


# -------------------------------------------------------------
# 8. 📊 MODEL PERFORMANCE & EXPLAINABILITY (XAI)
# -------------------------------------------------------------
elif nav_option == "📊 Model Performance & XAI":
    st.markdown("<div class='main-header'>MODEL PERFORMANCE & MACHINE LEARNING BENCHMARK</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Rigorous comparison across Linear Regression, Random Forest, Gradient Boosting, and XGBoost models.</div>", unsafe_allow_html=True)
    
    meta_path = os.path.join(BASE_DIR, "models", "model_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
            
        comp_dict = meta.get("model_comparison", {})
        metrics_rows = []
        for mdl_name, m_data in comp_dict.items():
            metrics_rows.append({
                "Model Algorithm": mdl_name,
                "MAE (Error)": m_data.get("MAE"),
                "MSE": m_data.get("MSE"),
                "RMSE": m_data.get("RMSE"),
                "R² Score": m_data.get("R2")
            })
            
        m_df = pd.DataFrame(metrics_rows).sort_values("R² Score", ascending=False)
        st.markdown("### 📈 Evaluation Metrics Comparison (Chronological Test Set)")
        st.dataframe(m_df, use_container_width=True)
        
        # Chart Comparison
        m_c1, m_c2 = st.columns(2)
        with m_c1:
            fig_r2 = px.bar(m_df, x="Model Algorithm", y="R² Score", color="Model Algorithm", title="R² Score (Higher is Better)", text="R² Score")
            fig_r2.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_r2, use_container_width=True)
        with m_c2:
            fig_rmse = px.bar(m_df, x="Model Algorithm", y="RMSE", color="Model Algorithm", title="Root Mean Squared Error (Lower is Better)", text="RMSE")
            fig_rmse.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_rmse, use_container_width=True)
            
        # Top Global Features
        best_name = meta.get("best_model_name", "Gradient Boosting")
        top_feats = comp_dict.get(best_name, {}).get("top_features", [])
        if top_feats:
            st.markdown(f"### 🌐 Global Feature Importance ({best_name})")
            feat_df = pd.DataFrame(top_feats)
            fig_glob = px.bar(
                feat_df,
                x="importance",
                y="feature",
                orientation="h",
                title=f"Top 12 Most Influential Predictive Features",
                labels={"importance": "Importance (%)", "feature": "Metric"}
            )
            fig_glob.update_layout(height=380, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_glob, use_container_width=True)


# -------------------------------------------------------------
# 9. 🗄️ SQL SCOUTING CONSOLE
# -------------------------------------------------------------
elif nav_option == "🗄️ SQL Scouting Console":
    st.markdown("<div class='main-header'>SQL SCOUTING & ANALYTICS CONSOLE</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Run structured SQL queries directly on the SQLite football performance database.</div>", unsafe_allow_html=True)
    
    preset_queries = {
        "Top 10 Rated Players in 2024/25": "SELECT p.player_name, p.position, p.club, p.league, s.player_rating, s.goals, s.assists, s.potential_score FROM performance_stats s JOIN players p ON s.player_id = p.player_id WHERE s.season = '2024-2025' ORDER BY s.player_rating DESC LIMIT 10;",
        "Top U-21 Elite Prospects": "SELECT p.player_name, s.age, p.position, p.club, s.potential_score, s.potential_tier, p.market_value FROM performance_stats s JOIN players p ON s.player_id = p.player_id WHERE s.season = '2024-2025' AND s.age <= 21 ORDER BY s.potential_score DESC LIMIT 10;",
        "Golden Boot Race (Goals Leaders)": "SELECT p.player_name, p.club, p.league, s.goals, s.assists, s.shots_on_target, s.attacking_score FROM performance_stats s JOIN players p ON s.player_id = p.player_id WHERE s.season = '2024-2025' ORDER BY s.goals DESC LIMIT 10;",
        "Top Playmakers by Chances & Passing": "SELECT p.player_name, p.position, p.club, s.key_passes, s.chances_created, s.pass_accuracy, s.passing_score FROM performance_stats s JOIN players p ON s.player_id = p.player_id WHERE s.season = '2024-2025' ORDER BY s.passing_score DESC LIMIT 10;",
        "Defensive Anchors (Tackles & Interceptions)": "SELECT p.player_name, p.club, s.tackles, s.interceptions, s.clearances, s.defensive_score FROM performance_stats s JOIN players p ON s.player_id = p.player_id WHERE s.season = '2024-2025' AND p.position = 'DF' ORDER BY s.defensive_score DESC LIMIT 10;"
    }
    
    sel_preset = st.selectbox("Select a Preset SQL Query", list(preset_queries.keys()))
    query_text = st.text_area("SQL Query Editor", value=preset_queries[sel_preset], height=120)
    
    if st.button("▶️ Execute SQL Query", use_container_width=True):
        try:
            res_df = db.run_query(query_text)
            st.success(f"Query returned {len(res_df)} rows.")
            st.dataframe(res_df, use_container_width=True)
        except Exception as e:
            st.error(f"SQL Error: {str(e)}")


# -------------------------------------------------------------
# 10. ℹ️ ABOUT & METHODOLOGY
# -------------------------------------------------------------
elif nav_option == "ℹ️ About Platform":
    st.markdown("<div class='main-header'>ABOUT FOOTBALL IQ</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>End-to-End Machine Learning & Football Intelligence Platform Architecture</div>", unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 System Objective
    **FOOTBALL IQ** is a production-grade football player performance prediction, potential valuation, and scouting intelligence platform designed for club analysts, sporting directors, and data scouts.
    
    ---
    ### 🔬 Methodological Architecture
    
    1. **Data Cleaning & Normalization**:
       - Position-specific imputation, strict range validation, and IQR outlier monitoring without truncating world-class performances.
    2. **Tactical Domain Engineering**:
       - *Attacking Score*: Weighted combination of goals, assists, shots on target, key passes, and chances created.
       - *Defensive Score*: Combination of tackles, interceptions, clearances, blocks, and duel recoveries.
       - *Passing & Progression Score*: Pass accuracy, key delivery, progressive passes, and successful 1v1 dribbles.
       - *Overall Performance Score*: Role-specific weighted composites (0 - 100).
    3. **Machine Learning Predictive Modeling**:
       - Chronological 80/20 train/test split preventing target leakage.
       - Regressors trained: Linear Regression, Random Forest, Gradient Boosting, XGBoost.
       - Models evaluate next-season rating trajectory with 95% RMSE confidence intervals.
    4. **AI Potential Scoring**:
       - Dynamic formula combining predicted rating trajectory (40%), current performance (25%), age curve factor (20%), and minutes played (15%).
    5. **Unsupervised Player Archetypes**:
       - K-Means clustering (k=5) with 2D/3D PCA projection identifying player playstyles.
    6. **Explainable AI (XAI)**:
       - Feature importances and local factor attribution explaining every prediction.
    """)
