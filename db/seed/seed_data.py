import os
import json
import math
import random
import argparse
from typing import List, Dict, Any, Tuple
import numpy as np

# Reproducibility seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

LEAGUES_CONFIG = {
    "Premier League": {"tier": 1, "coeff": 1.50, "country": "England"},
    "La Liga": {"tier": 1, "coeff": 1.30, "country": "Spain"},
    "Bundesliga": {"tier": 1, "coeff": 1.15, "country": "Germany"},
    "Serie A": {"tier": 1, "coeff": 1.15, "country": "Italy"},
    "Ligue 1": {"tier": 1, "coeff": 1.05, "country": "France"},
    "Liga Portugal": {"tier": 2, "coeff": 0.75, "country": "Portugal"},
    "Eredivisie": {"tier": 2, "coeff": 0.70, "country": "Netherlands"},
    "Belgian Pro League": {"tier": 2, "coeff": 0.60, "country": "Belgium"},
    "Brasileirão": {"tier": 2, "coeff": 0.55, "country": "Brazil"},
    "Championship": {"tier": 2, "coeff": 0.65, "country": "England"},
}

TEAMS_BY_LEAGUE = {
    "Premier League": ["Manchester City", "Arsenal", "Liverpool", "Aston Villa", "Tottenham", "Newcastle", "Chelsea", "Brighton", "West Ham", "Bournemouth"],
    "La Liga": ["Real Madrid", "Barcelona", "Atletico Madrid", "Real Sociedad", "Athletic Club", "Girona", "Villarreal", "Real Betis"],
    "Bundesliga": ["Bayer Leverkusen", "Bayern Munich", "RB Leipzig", "Borussia Dortmund", "Stuttgart", "Eintracht Frankfurt"],
    "Serie A": ["Inter Milan", "AC Milan", "Juventus", "Atalanta", "AS Roma", "Napoli", "Lazio", "Fiorentina"],
    "Ligue 1": ["Paris Saint-Germain", "Monaco", "Lille", "Brest", "Nice", "Marseille", "Lyon"],
    "Liga Portugal": ["Sporting CP", "Benfica", "FC Porto", "SC Braga", "Vitoria SC"],
    "Eredivisie": ["PSV Eindhoven", "Feyenoord", "Ajax", "AZ Alkmaar", "FC Twente"],
    "Belgian Pro League": ["Club Brugge", "Union Saint-Gilloise", "Anderlecht", "Genk", "Gent"],
    "Brasileirão": ["Palmeiras", "Flamengo", "Botafogo", "Atletico Mineiro", "Sao Paulo", "Fluminense"],
    "Championship": ["Leeds United", "Leicester City", "Southampton", "Ipswich Town", "Norwich City", "West Bromwich"],
}

# Role distribution parameters (Mean, Std) for p90 metrics
ROLE_METRIC_PROFILES = {
    "CB": {
        "npxg_p90": (0.04, 0.02, 0.0, 0.12),
        "xag_p90": (0.03, 0.02, 0.0, 0.10),
        "shots_total_p90": (0.45, 0.20, 0.1, 1.2),
        "shots_on_target_pct": (28.0, 10.0, 10.0, 60.0),
        "shot_creating_actions_p90": (0.85, 0.35, 0.2, 2.0),
        "goal_creating_actions_p90": (0.06, 0.04, 0.0, 0.25),
        "key_passes_p90": (0.22, 0.12, 0.05, 0.65),
        "pass_completion_pct": (88.5, 3.5, 78.0, 96.0),
        "passes_attempted_p90": (62.0, 12.0, 38.0, 92.0),
        "prgp_p90": (3.8, 1.3, 1.5, 7.5),
        "passes_into_final_third_p90": (3.2, 1.2, 1.0, 6.5),
        "passes_into_penalty_area_p90": (0.15, 0.10, 0.0, 0.55),
        "prgc_p90": (0.95, 0.45, 0.2, 2.6),
        "dribbles_completed_p90": (0.24, 0.14, 0.05, 0.75),
        "dribble_success_pct": (68.0, 15.0, 30.0, 95.0),
        "carries_into_final_third_p90": (0.35, 0.20, 0.05, 1.1),
        "dispossessed_p90": (0.30, 0.15, 0.05, 0.8),
        "tackles_p90": (2.1, 0.55, 1.0, 3.8),
        "tackles_won_p90": (1.45, 0.40, 0.6, 2.8),
        "interceptions_p90": (1.75, 0.50, 0.8, 3.2),
        "blocks_p90": (1.35, 0.45, 0.5, 2.6),
        "clearances_p90": (4.8, 1.3, 2.2, 8.5),
        "recoveries_p90": (5.8, 1.3, 3.2, 9.5),
        "tackles_def_3rd_p90": (1.55, 0.45, 0.7, 2.9),
        "tackles_mid_3rd_p90": (0.50, 0.20, 0.15, 1.1),
        "tackles_att_3rd_p90": (0.05, 0.03, 0.0, 0.15),
        "aerial_duels_won_pct": (65.0, 7.5, 45.0, 85.0),
        "aerial_duels_total_p90": (3.8, 1.3, 1.8, 7.2),
        "goals_p90": (0.04, 0.03, 0.0, 0.15),
        "assists_p90": (0.03, 0.02, 0.0, 0.12),
    },
    "FB": {
        "npxg_p90": (0.06, 0.03, 0.01, 0.18),
        "xag_p90": (0.16, 0.07, 0.04, 0.38),
        "shots_total_p90": (0.75, 0.35, 0.2, 2.0),
        "shots_on_target_pct": (29.0, 12.0, 10.0, 60.0),
        "shot_creating_actions_p90": (2.3, 0.75, 1.0, 4.5),
        "goal_creating_actions_p90": (0.22, 0.11, 0.05, 0.6),
        "key_passes_p90": (1.25, 0.45, 0.4, 2.8),
        "pass_completion_pct": (81.0, 5.0, 68.0, 92.0),
        "passes_attempted_p90": (58.0, 12.0, 35.0, 88.0),
        "prgp_p90": (4.9, 1.4, 2.2, 8.8),
        "passes_into_final_third_p90": (3.6, 1.2, 1.5, 7.0),
        "passes_into_penalty_area_p90": (1.45, 0.55, 0.4, 3.2),
        "prgc_p90": (2.8, 1.1, 1.0, 6.0),
        "dribbles_completed_p90": (1.1, 0.5, 0.3, 2.8),
        "dribble_success_pct": (54.0, 12.0, 25.0, 80.0),
        "carries_into_final_third_p90": (1.6, 0.65, 0.5, 3.8),
        "dispossessed_p90": (0.85, 0.30, 0.25, 1.8),
        "tackles_p90": (2.6, 0.7, 1.2, 4.5),
        "tackles_won_p90": (1.7, 0.5, 0.7, 3.2),
        "interceptions_p90": (1.4, 0.45, 0.5, 2.8),
        "blocks_p90": (0.9, 0.35, 0.3, 2.0),
        "clearances_p90": (2.4, 0.8, 0.8, 4.8),
        "recoveries_p90": (6.4, 1.4, 3.8, 10.5),
        "tackles_def_3rd_p90": (1.4, 0.45, 0.6, 2.8),
        "tackles_mid_3rd_p90": (0.95, 0.35, 0.3, 2.0),
        "tackles_att_3rd_p90": (0.25, 0.12, 0.05, 0.6),
        "aerial_duels_won_pct": (52.0, 9.0, 30.0, 75.0),
        "aerial_duels_total_p90": (2.2, 0.85, 0.8, 4.8),
        "goals_p90": (0.05, 0.04, 0.0, 0.2),
        "assists_p90": (0.14, 0.07, 0.02, 0.35),
    },
    "CM": {
        "npxg_p90": (0.10, 0.05, 0.02, 0.26),
        "xag_p90": (0.15, 0.06, 0.03, 0.35),
        "shots_total_p90": (1.2, 0.5, 0.3, 2.6),
        "shots_on_target_pct": (31.0, 11.0, 10.0, 60.0),
        "shot_creating_actions_p90": (2.8, 0.85, 1.2, 5.2),
        "goal_creating_actions_p90": (0.25, 0.12, 0.05, 0.65),
        "key_passes_p90": (1.35, 0.50, 0.4, 3.0),
        "pass_completion_pct": (86.5, 4.5, 74.0, 95.0),
        "passes_attempted_p90": (68.0, 14.0, 42.0, 105.0),
        "prgp_p90": (6.2, 1.8, 2.8, 11.0),
        "passes_into_final_third_p90": (5.2, 1.6, 2.2, 9.5),
        "passes_into_penalty_area_p90": (1.1, 0.45, 0.3, 2.6),
        "prgc_p90": (1.8, 0.8, 0.6, 4.2),
        "dribbles_completed_p90": (0.85, 0.45, 0.2, 2.4),
        "dribble_success_pct": (62.0, 13.0, 30.0, 88.0),
        "carries_into_final_third_p90": (1.2, 0.55, 0.3, 3.0),
        "dispossessed_p90": (0.95, 0.35, 0.3, 2.0),
        "tackles_p90": (2.5, 0.75, 1.1, 4.8),
        "tackles_won_p90": (1.65, 0.55, 0.6, 3.4),
        "interceptions_p90": (1.35, 0.45, 0.4, 2.8),
        "blocks_p90": (0.95, 0.35, 0.3, 2.0),
        "clearances_p90": (1.3, 0.55, 0.4, 3.0),
        "recoveries_p90": (7.2, 1.6, 4.0, 12.0),
        "tackles_def_3rd_p90": (0.9, 0.35, 0.3, 2.1),
        "tackles_mid_3rd_p90": (1.35, 0.45, 0.5, 2.8),
        "tackles_att_3rd_p90": (0.25, 0.12, 0.05, 0.65),
        "aerial_duels_won_pct": (54.0, 10.0, 28.0, 78.0),
        "aerial_duels_total_p90": (2.1, 0.9, 0.6, 4.8),
        "goals_p90": (0.09, 0.05, 0.01, 0.24),
        "assists_p90": (0.13, 0.06, 0.02, 0.32),
    },
    "CAM": {
        "npxg_p90": (0.22, 0.08, 0.07, 0.45),
        "xag_p90": (0.32, 0.10, 0.12, 0.62),
        "shots_total_p90": (2.2, 0.7, 0.8, 4.2),
        "shots_on_target_pct": (36.0, 10.0, 15.0, 65.0),
        "shot_creating_actions_p90": (4.6, 1.2, 2.2, 7.8),
        "goal_creating_actions_p90": (0.55, 0.20, 0.15, 1.2),
        "key_passes_p90": (2.4, 0.75, 1.0, 4.5),
        "pass_completion_pct": (79.0, 5.0, 66.0, 90.0),
        "passes_attempted_p90": (48.0, 11.0, 28.0, 75.0),
        "prgp_p90": (5.4, 1.5, 2.5, 9.2),
        "passes_into_final_third_p90": (4.0, 1.3, 1.8, 7.5),
        "passes_into_penalty_area_p90": (2.3, 0.8, 0.8, 4.8),
        "prgc_p90": (3.2, 1.2, 1.2, 6.8),
        "dribbles_completed_p90": (1.6, 0.7, 0.5, 3.8),
        "dribble_success_pct": (55.0, 12.0, 25.0, 82.0),
        "carries_into_final_third_p90": (2.4, 0.9, 0.8, 5.0),
        "dispossessed_p90": (1.6, 0.5, 0.6, 3.0),
        "tackles_p90": (1.4, 0.45, 0.5, 2.8),
        "tackles_won_p90": (0.85, 0.30, 0.25, 1.8),
        "interceptions_p90": (0.65, 0.25, 0.15, 1.4),
        "blocks_p90": (0.55, 0.20, 0.15, 1.2),
        "clearances_p90": (0.6, 0.3, 0.1, 1.5),
        "recoveries_p90": (4.8, 1.2, 2.2, 8.0),
        "tackles_def_3rd_p90": (0.35, 0.15, 0.1, 0.85),
        "tackles_mid_3rd_p90": (0.75, 0.30, 0.25, 1.6),
        "tackles_att_3rd_p90": (0.30, 0.15, 0.05, 0.75),
        "aerial_duels_won_pct": (42.0, 12.0, 18.0, 68.0),
        "aerial_duels_total_p90": (1.4, 0.65, 0.3, 3.2),
        "goals_p90": (0.21, 0.09, 0.05, 0.48),
        "assists_p90": (0.28, 0.10, 0.08, 0.58),
    },
    "W": {
        "npxg_p90": (0.28, 0.10, 0.10, 0.58),
        "xag_p90": (0.26, 0.09, 0.08, 0.54),
        "shots_total_p90": (2.6, 0.8, 1.1, 4.8),
        "shots_on_target_pct": (38.0, 9.0, 18.0, 65.0),
        "shot_creating_actions_p90": (4.2, 1.1, 2.0, 7.2),
        "goal_creating_actions_p90": (0.50, 0.18, 0.12, 1.1),
        "key_passes_p90": (1.9, 0.65, 0.8, 3.8),
        "pass_completion_pct": (76.0, 5.5, 62.0, 88.0),
        "passes_attempted_p90": (36.0, 9.0, 20.0, 58.0),
        "prgp_p90": (3.4, 1.2, 1.4, 6.8),
        "passes_into_final_third_p90": (2.1, 0.8, 0.7, 4.5),
        "passes_into_penalty_area_p90": (1.9, 0.7, 0.6, 4.0),
        "prgc_p90": (4.8, 1.6, 2.0, 9.5),
        "dribbles_completed_p90": (2.5, 1.0, 0.8, 5.8),
        "dribble_success_pct": (52.0, 11.0, 26.0, 78.0),
        "carries_into_final_third_p90": (3.2, 1.1, 1.2, 6.5),
        "dispossessed_p90": (1.9, 0.6, 0.7, 3.6),
        "tackles_p90": (1.5, 0.5, 0.5, 3.0),
        "tackles_won_p90": (0.95, 0.35, 0.3, 2.0),
        "interceptions_p90": (0.6, 0.25, 0.15, 1.4),
        "blocks_p90": (0.5, 0.2, 0.15, 1.1),
        "clearances_p90": (0.45, 0.25, 0.1, 1.2),
        "recoveries_p90": (4.5, 1.2, 2.2, 7.8),
        "tackles_def_3rd_p90": (0.4, 0.2, 0.1, 1.0),
        "tackles_mid_3rd_p90": (0.75, 0.3, 0.2, 1.6),
        "tackles_att_3rd_p90": (0.35, 0.15, 0.1, 0.85),
        "aerial_duels_won_pct": (38.0, 12.0, 15.0, 65.0),
        "aerial_duels_total_p90": (1.2, 0.6, 0.3, 3.0),
        "goals_p90": (0.27, 0.11, 0.08, 0.60),
        "assists_p90": (0.24, 0.09, 0.06, 0.52),
    },
    "ST": {
        "npxg_p90": (0.52, 0.16, 0.20, 0.98),
        "xag_p90": (0.16, 0.07, 0.04, 0.38),
        "shots_total_p90": (3.4, 0.9, 1.6, 6.0),
        "shots_on_target_pct": (44.0, 9.0, 22.0, 68.0),
        "shot_creating_actions_p90": (2.6, 0.8, 1.1, 5.0),
        "goal_creating_actions_p90": (0.40, 0.16, 0.1, 0.9),
        "key_passes_p90": (1.1, 0.45, 0.3, 2.5),
        "pass_completion_pct": (72.0, 6.0, 56.0, 85.0),
        "passes_attempted_p90": (22.0, 6.5, 12.0, 40.0),
        "prgp_p90": (1.9, 0.75, 0.6, 4.2),
        "passes_into_final_third_p90": (1.2, 0.5, 0.4, 2.8),
        "passes_into_penalty_area_p90": (0.9, 0.4, 0.2, 2.2),
        "prgc_p90": (2.1, 0.9, 0.6, 4.8),
        "dribbles_completed_p90": (0.95, 0.45, 0.25, 2.5),
        "dribble_success_pct": (48.0, 12.0, 22.0, 76.0),
        "carries_into_final_third_p90": (1.4, 0.6, 0.4, 3.2),
        "dispossessed_p90": (1.7, 0.5, 0.6, 3.2),
        "tackles_p90": (0.8, 0.35, 0.2, 2.0),
        "tackles_won_p90": (0.45, 0.2, 0.1, 1.2),
        "interceptions_p90": (0.3, 0.15, 0.05, 0.8),
        "blocks_p90": (0.4, 0.2, 0.1, 1.0),
        "clearances_p90": (0.6, 0.3, 0.1, 1.6),
        "recoveries_p90": (2.8, 0.9, 1.2, 5.5),
        "tackles_def_3rd_p90": (0.1, 0.08, 0.0, 0.35),
        "tackles_mid_3rd_p90": (0.35, 0.18, 0.08, 0.9),
        "tackles_att_3rd_p90": (0.35, 0.18, 0.08, 0.95),
        "aerial_duels_won_pct": (46.0, 12.0, 20.0, 74.0),
        "aerial_duels_total_p90": (3.2, 1.4, 1.0, 7.5),
        "goals_p90": (0.50, 0.18, 0.18, 1.05),
        "assists_p90": (0.14, 0.07, 0.02, 0.36),
    },
}

ANCHOR_STARS = [
    {
        "name": "Rodri",
        "age": 29,
        "nationality": "Spain",
        "club": "Manchester City",
        "league": "Premier League",
        "primary_position": "CM",
        "secondary_position": "CB",
        "market_value_eur": 110_000_000,
        "contract_expires_year": 2027,
        "minutes_played": 2850,
        "matches_played": 34,
        "stats_overrides": {
            "pass_completion_pct": 92.8,
            "passes_attempted_p90": 98.4,
            "prgp_p90": 10.4,
            "passes_into_final_third_p90": 9.8,
            "recoveries_p90": 8.9,
            "tackles_p90": 2.8,
            "tackles_won_p90": 1.9,
            "interceptions_p90": 1.6,
            "npxg_p90": 0.14,
            "xag_p90": 0.18,
            "goals_p90": 0.16,
            "assists_p90": 0.19,
        },
    },
    {
        "name": "Vinicius Jr",
        "age": 26,
        "nationality": "Brazil",
        "club": "Real Madrid",
        "league": "La Liga",
        "primary_position": "W",
        "secondary_position": "ST",
        "market_value_eur": 160_000_000,
        "contract_expires_year": 2028,
        "minutes_played": 2600,
        "matches_played": 31,
        "stats_overrides": {
            "prgc_p90": 8.6,
            "dribbles_completed_p90": 4.8,
            "carries_into_final_third_p90": 5.4,
            "npxg_p90": 0.48,
            "xag_p90": 0.36,
            "goals_p90": 0.52,
            "assists_p90": 0.32,
            "shot_creating_actions_p90": 6.2,
        },
    },
    {
        "name": "Bukayo Saka",
        "age": 25,
        "nationality": "England",
        "club": "Arsenal",
        "league": "Premier League",
        "primary_position": "W",
        "secondary_position": "CAM",
        "market_value_eur": 130_000_000,
        "contract_expires_year": 2028,
        "minutes_played": 2750,
        "matches_played": 33,
        "stats_overrides": {
            "prgc_p90": 6.8,
            "dribbles_completed_p90": 3.4,
            "npxg_p90": 0.38,
            "xag_p90": 0.42,
            "key_passes_p90": 2.9,
            "shot_creating_actions_p90": 5.8,
            "goals_p90": 0.42,
            "assists_p90": 0.38,
        },
    },
    {
        "name": "William Saliba",
        "age": 25,
        "nationality": "France",
        "club": "Arsenal",
        "league": "Premier League",
        "primary_position": "CB",
        "secondary_position": None,
        "market_value_eur": 85_000_000,
        "contract_expires_year": 2028,
        "minutes_played": 3000,
        "matches_played": 35,
        "stats_overrides": {
            "pass_completion_pct": 93.4,
            "passes_attempted_p90": 74.0,
            "prgp_p90": 4.8,
            "tackles_won_p90": 1.7,
            "interceptions_p90": 1.8,
            "aerial_duels_won_pct": 72.0,
            "clearances_p90": 4.2,
        },
    },
    {
        "name": "Florian Wirtz",
        "age": 23,
        "nationality": "Germany",
        "club": "Bayer Leverkusen",
        "league": "Bundesliga",
        "primary_position": "CAM",
        "secondary_position": "W",
        "market_value_eur": 130_000_000,
        "contract_expires_year": 2027,
        "minutes_played": 2550,
        "matches_played": 30,
        "stats_overrides": {
            "xag_p90": 0.48,
            "key_passes_p90": 3.4,
            "shot_creating_actions_p90": 6.8,
            "goal_creating_actions_p90": 0.95,
            "prgp_p90": 7.4,
            "prgc_p90": 5.2,
            "goals_p90": 0.38,
            "assists_p90": 0.44,
        },
    },
    {
        "name": "Erling Haaland",
        "age": 26,
        "nationality": "Norway",
        "club": "Manchester City",
        "league": "Premier League",
        "primary_position": "ST",
        "secondary_position": None,
        "market_value_eur": 180_000_000,
        "contract_expires_year": 2028,
        "minutes_played": 2700,
        "matches_played": 32,
        "stats_overrides": {
            "npxg_p90": 0.88,
            "goals_p90": 0.92,
            "shots_total_p90": 4.4,
            "shots_on_target_pct": 52.0,
            "aerial_duels_won_pct": 56.0,
        },
    },
]

# Donor league Moneyball gems (Analog anchors with Rodri/Vinicius metrics at discounted fee)
DONOR_GEMS = [
    {
        "name": "Marten Hjulmand",
        "age": 26,
        "nationality": "Denmark",
        "club": "Sporting CP",
        "league": "Liga Portugal",
        "primary_position": "CM",
        "secondary_position": "CB",
        "market_value_eur": 35_000_000,  # Arbitrage vs €110M Rodri!
        "contract_expires_year": 2028,
        "minutes_played": 2500,
        "matches_played": 30,
        "stats_overrides": {
            "pass_completion_pct": 90.2,
            "passes_attempted_p90": 88.0,
            "prgp_p90": 8.8,
            "passes_into_final_third_p90": 8.2,
            "recoveries_p90": 8.4,
            "tackles_p90": 2.7,
            "tackles_won_p90": 1.8,
            "interceptions_p90": 1.7,
        },
    },
    {
        "name": "Alan Varela",
        "age": 25,
        "nationality": "Argentina",
        "club": "FC Porto",
        "league": "Liga Portugal",
        "primary_position": "CM",
        "secondary_position": None,
        "market_value_eur": 32_000_000,
        "contract_expires_year": 2028,
        "minutes_played": 2400,
        "matches_played": 29,
        "stats_overrides": {
            "pass_completion_pct": 89.5,
            "passes_attempted_p90": 82.0,
            "prgp_p90": 7.9,
            "recoveries_p90": 7.8,
            "tackles_p90": 2.9,
            "tackles_won_p90": 1.9,
            "interceptions_p90": 1.6,
        },
    },
    {
        "name": "Johan Bakayoko",
        "age": 23,
        "nationality": "Belgium",
        "club": "PSV Eindhoven",
        "league": "Eredivisie",
        "primary_position": "W",
        "secondary_position": "CAM",
        "market_value_eur": 38_000_000,  # Arbitrage vs €130M Saka
        "contract_expires_year": 2027,
        "minutes_played": 2350,
        "matches_played": 28,
        "stats_overrides": {
            "prgc_p90": 6.2,
            "dribbles_completed_p90": 3.6,
            "npxg_p90": 0.35,
            "xag_p90": 0.40,
            "key_passes_p90": 2.7,
            "shot_creating_actions_p90": 5.4,
            "goals_p90": 0.36,
            "assists_p90": 0.35,
        },
    },
    {
        "name": "Jorrel Hato",
        "age": 20,
        "nationality": "Netherlands",
        "club": "Ajax",
        "league": "Eredivisie",
        "primary_position": "CB",
        "secondary_position": "FB",
        "market_value_eur": 28_000_000,  # Wonderkid arbitrage vs €85M Saliba
        "contract_expires_year": 2028,
        "minutes_played": 2700,
        "matches_played": 32,
        "stats_overrides": {
            "pass_completion_pct": 91.5,
            "passes_attempted_p90": 78.0,
            "prgp_p90": 5.2,
            "prgc_p90": 1.8,
            "tackles_won_p90": 1.8,
            "interceptions_p90": 1.7,
            "aerial_duels_won_pct": 68.0,
        },
    },
]

FIRST_NAMES = [
    "Lucas", "Mateo", "Gabriel", "Enzo", "Felipe", "Thiago", "Arthur", "Leo", "Noah",
    "Jonas", "Max", "Luka", "Sandro", "Milos", "Julian", "Alexander", "David", "Viktor",
    "Daniel", "Hugo", "Nicolas", "Federico", "Santiago", "Valentin", "Rafael", "Diogo",
    "Joao", "Pedro", "Goncalo", "Nuno", "Benoit", "Florian", "Amine", "Bilal", "Youssef"
]

LAST_NAMES = [
    "Silva", "Santos", "Fernandes", "Rodriguez", "Gomez", "Lopez", "Pereira", "Costa",
    "Martins", "Almeida", "Schmidt", "Mueller", "Fischer", "Weber", "Meyer", "Jorgensen",
    "Nielsen", "Larsson", "Andersson", "Berg", "Moreno", "Navarro", "Alvarez", "Castillo",
    "Diallo", "Traore", "Camara", "Kone", "Mensah", "Dembele", "Benali", "Bouzid"
]


def generate_player_metrics(position: str, overrides: Dict[str, float] = None) -> Dict[str, float]:
    """Generates realistic correlated p90 stats for a specific tactical position."""
    profile = ROLE_METRIC_PROFILES[position]
    metrics = {}
    for metric_name, (mean, std, min_val, max_val) in profile.items():
        val = np.random.normal(mean, std)
        val = float(np.clip(val, min_val, max_val))
        metrics[metric_name] = round(val, 2)
    if overrides:
        for k, v in overrides.items():
            if k in metrics:
                metrics[k] = round(float(v), 2)
    return metrics


def calculate_performance_score(position: str, stats: Dict[str, float]) -> float:
    """Calculates a normalized 0-100 composite performance score based on role weights."""
    if position == "CB":
        score = (
            stats["aerial_duels_won_pct"] * 0.25 +
            stats["pass_completion_pct"] * 0.25 +
            (stats["tackles_won_p90"] / 2.5 * 100) * 0.20 +
            (stats["interceptions_p90"] / 2.5 * 100) * 0.15 +
            (stats["clearances_p90"] / 6.0 * 100) * 0.15
        )
    elif position == "FB":
        score = (
            (stats["prgp_p90"] / 8.0 * 100) * 0.25 +
            (stats["prgc_p90"] / 6.0 * 100) * 0.25 +
            (stats["tackles_p90"] / 4.0 * 100) * 0.20 +
            (stats["xag_p90"] / 0.35 * 100) * 0.15 +
            (stats["recoveries_p90"] / 9.0 * 100) * 0.15
        )
    elif position == "CM":
        score = (
            (stats["prgp_p90"] / 9.0 * 100) * 0.30 +
            stats["pass_completion_pct"] * 0.25 +
            (stats["recoveries_p90"] / 10.0 * 100) * 0.25 +
            (stats["tackles_p90"] / 4.0 * 100) * 0.20
        )
    elif position == "CAM":
        score = (
            (stats["xag_p90"] / 0.50 * 100) * 0.35 +
            (stats["shot_creating_actions_p90"] / 7.0 * 100) * 0.30 +
            (stats["key_passes_p90"] / 3.5 * 100) * 0.20 +
            (stats["npxg_p90"] / 0.40 * 100) * 0.15
        )
    elif position == "W":
        score = (
            (stats["prgc_p90"] / 7.5 * 100) * 0.30 +
            (stats["dribbles_completed_p90"] / 4.5 * 100) * 0.25 +
            (stats["npxg_p90"] / 0.50 * 100) * 0.25 +
            (stats["xag_p90"] / 0.40 * 100) * 0.20
        )
    else:  # ST
        score = (
            (stats["npxg_p90"] / 0.80 * 100) * 0.45 +
            (stats["shots_on_target_pct"]) * 0.25 +
            (stats["shots_total_p90"] / 4.5 * 100) * 0.15 +
            (stats["goals_p90"] / 0.85 * 100) * 0.15
        )
    return round(float(np.clip(score, 30.0, 99.0)), 1)


def generate_market_value(
    age: int,
    league: str,
    position: str,
    perf_score: float,
    minutes: int,
    is_anchor: bool = False,
    anchor_value: float = None,
) -> float:
    """
    Logarithmic realistic market value generator respecting age curves and league premiums.
    Young talents (19-23) with elite stats command high multipliers; older players decay.
    """
    if is_anchor and anchor_value is not None:
        return float(anchor_value)

    league_coeff = LEAGUES_CONFIG[league]["coeff"]
    
    # Age curve: peak value at 23-26, youth hype at 19-21, decay after 28
    if age <= 21:
        age_multiplier = 1.35
    elif age <= 25:
        age_multiplier = 1.20
    elif age <= 28:
        age_multiplier = 1.00
    elif age <= 31:
        age_multiplier = 0.70
    else:
        age_multiplier = 0.40

    # Minutes stability
    min_factor = np.clip(minutes / 2200.0, 0.4, 1.2)

    # Base price exponential in performance
    normalized_perf = (perf_score - 50.0) / 10.0  # z-score-like
    base_val = 12_000_000 * math.exp(0.48 * normalized_perf)
    
    # Combined valuation with realistic market noise
    noise = np.random.uniform(0.85, 1.15)
    raw_val = base_val * league_coeff * age_multiplier * min_factor * noise

    # Clip within realistic market bounds
    val = float(np.clip(raw_val, 800_000, 180_000_000))
    # Round to hundreds of thousands
    return round(val / 100_000) * 100_000


def generate_tactical_embedding(position: str, stats: Dict[str, float]) -> List[float]:
    """
    Builds a 64-dimensional normalized tactical style embedding vector.
    First 30 dims map normalized p90 metrics; remaining dims encode tactical nuance.
    """
    raw_features = [
        stats["npxg_p90"] / 1.0,
        stats["xag_p90"] / 0.6,
        stats["shots_total_p90"] / 5.0,
        stats["shots_on_target_pct"] / 100.0,
        stats["shot_creating_actions_p90"] / 8.0,
        stats["goal_creating_actions_p90"] / 1.2,
        stats["key_passes_p90"] / 4.0,
        stats["pass_completion_pct"] / 100.0,
        stats["passes_attempted_p90"] / 100.0,
        stats["prgp_p90"] / 10.0,
        stats["passes_into_final_third_p90"] / 8.0,
        stats["passes_into_penalty_area_p90"] / 4.0,
        stats["prgc_p90"] / 8.0,
        stats["dribbles_completed_p90"] / 5.0,
        stats["dribble_success_pct"] / 100.0,
        stats["carries_into_final_third_p90"] / 5.0,
        stats["dispossessed_p90"] / 3.0,
        stats["tackles_p90"] / 5.0,
        stats["tackles_won_p90"] / 3.5,
        stats["interceptions_p90"] / 3.0,
        stats["blocks_p90"] / 2.5,
        stats["clearances_p90"] / 8.0,
        stats["recoveries_p90"] / 11.0,
        stats["tackles_def_3rd_p90"] / 3.0,
        stats["tackles_mid_3rd_p90"] / 3.0,
        stats["tackles_att_3rd_p90"] / 1.0,
        stats["aerial_duels_won_pct"] / 100.0,
        stats["aerial_duels_total_p90"] / 7.0,
        stats["goals_p90"] / 1.0,
        stats["assists_p90"] / 0.6,
    ]

    # Deterministic position one-hot signature
    pos_idx = {"CB": 0, "FB": 1, "CM": 2, "CAM": 3, "W": 4, "ST": 5}[position]
    pos_encoding = [0.0] * 6
    pos_encoding[pos_idx] = 1.0

    # Project to 64 dims with deterministic transformation
    combined = raw_features + pos_encoding  # 36 dims
    extra_dims = [math.sin((i + 1) * sum(combined[:10])) * 0.5 for i in range(28)]  # 28 dims -> total 64
    full_vector = np.array(combined + extra_dims, dtype=np.float32)

    # L2 Normalization (critical for cosine similarity)
    norm = np.linalg.norm(full_vector)
    if norm > 0:
        full_vector = full_vector / norm
    return [round(float(x), 6) for x in full_vector.tolist()]


def generate_full_dataset(target_count: int = 550) -> List[Dict[str, Any]]:
    """Generates full 550+ dataset with anchor stars, hidden donor-league gems and synthetic players."""
    players = []
    player_id = 1

    # 1. Add Anchor Stars
    for star in ANCHOR_STARS:
        pos = star["primary_position"]
        stats = generate_player_metrics(pos, star.get("stats_overrides"))
        perf = calculate_performance_score(pos, stats)
        emb = generate_tactical_embedding(pos, stats)
        actual_val = star["market_value_eur"]

        # Fair predicted value (stars are accurately/fully priced)
        pred_val = actual_val * random.uniform(0.95, 1.08)
        delta = pred_val - actual_val

        p_record = {
            "id": player_id,
            "name": star["name"],
            "age": star["age"],
            "nationality": star["nationality"],
            "club": star["club"],
            "league": star["league"],
            "primary_position": pos,
            "secondary_position": star["secondary_position"],
            "market_value_eur": actual_val,
            "contract_expires_year": star["contract_expires_year"],
            "minutes_played": star["minutes_played"],
            "matches_played": star["matches_played"],
            "performance_score": perf,
            "stats": stats,
            "embedding": emb,
            "valuation": {
                "predicted_value_eur": round(pred_val),
                "actual_value_eur": round(actual_val),
                "undervaluation_delta_eur": round(delta),
                "undervaluation_ratio": round(pred_val / (actual_val + 1), 3),
                "confidence_score": 0.92,
                "shap_drivers": [
                    {"feature": "prgp_p90", "impact_eur": 18_000_000, "direction": "positive"},
                    {"feature": "league_tier", "impact_eur": 25_000_000, "direction": "positive"},
                    {"feature": "age_curve", "impact_eur": -4_000_000, "direction": "negative"},
                ],
            },
        }
        players.append(p_record)
        player_id += 1

    # 2. Add Donor Gems (High performance, donor league discount -> High Moneyball Delta!)
    for gem in DONOR_GEMS:
        pos = gem["primary_position"]
        stats = generate_player_metrics(pos, gem.get("stats_overrides"))
        perf = calculate_performance_score(pos, stats)
        emb = generate_tactical_embedding(pos, stats)
        actual_val = gem["market_value_eur"]

        # Model estimates fair value based on metrics matching Tier 1 stars:
        # e.g., Hjulmand (€35M actual) has predicted value €65M (+€30M delta!)
        pred_val = actual_val * random.uniform(1.60, 2.10)
        delta = pred_val - actual_val

        p_record = {
            "id": player_id,
            "name": gem["name"],
            "age": gem["age"],
            "nationality": gem["nationality"],
            "club": gem["club"],
            "league": gem["league"],
            "primary_position": pos,
            "secondary_position": gem["secondary_position"],
            "market_value_eur": actual_val,
            "contract_expires_year": gem["contract_expires_year"],
            "minutes_played": gem["minutes_played"],
            "matches_played": gem["matches_played"],
            "performance_score": perf,
            "stats": stats,
            "embedding": emb,
            "valuation": {
                "predicted_value_eur": round(pred_val),
                "actual_value_eur": round(actual_val),
                "undervaluation_delta_eur": round(delta),
                "undervaluation_ratio": round(pred_val / (actual_val + 1), 3),
                "confidence_score": 0.88,
                "shap_drivers": [
                    {"feature": "prgp_p90", "impact_eur": 14_500_000, "direction": "positive"},
                    {"feature": "recoveries_p90", "impact_eur": 9_200_000, "direction": "positive"},
                    {"feature": "league_discount", "impact_eur": -12_000_000, "direction": "negative"},
                ],
            },
        }
        players.append(p_record)
        player_id += 1

    # 3. Generate Remaining Synthetic Players to reach target_count (550+)
    positions_pool = ["CB", "FB", "CM", "CAM", "W", "ST"]
    leagues_pool = list(LEAGUES_CONFIG.keys())
    existing_names = set(p["name"] for p in players)

    while len(players) < target_count:
        pos = random.choice(positions_pool)
        league = random.choice(leagues_pool)
        club = random.choice(TEAMS_BY_LEAGUE[league])
        nat = random.choice(["Spain", "France", "Brazil", "Germany", "England", "Portugal", "Netherlands", "Argentina", "Belgium", "Denmark", "Uruguay", "Colombia", "Nigeria"])
        
        # Name generation
        fname = random.choice(FIRST_NAMES)
        lname = random.choice(LAST_NAMES)
        full_name = f"{fname} {lname}"
        if full_name in existing_names:
            full_name = f"{full_name} Jr."
        existing_names.add(full_name)

        ages_pool = list(range(18, 34))  # 18 to 33 inclusive (16 elements)
        # Higher density around 21-26
        age_probs = [0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.12, 0.10, 0.08, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.005]
        age = int(np.random.choice(ages_pool, p=age_probs))
        minutes = int(random.randint(950, 3100))
        matches = int(round(minutes / 85))
        contract = random.choice([2026, 2027, 2028, 2029])

        stats = generate_player_metrics(pos)
        perf = calculate_performance_score(pos, stats)
        emb = generate_tactical_embedding(pos, stats)
        actual_val = generate_market_value(age, league, pos, perf, minutes)

        # Baseline predicted fair valuation (derived by sport metrics)
        # In donor leagues or young high performers, model recognizes undervaluation
        is_undervalued = (LEAGUES_CONFIG[league]["tier"] == 2 and perf >= 72.0) or (age <= 22 and perf >= 75.0)
        if is_undervalued:
            multiplier = random.uniform(1.25, 1.85)
        elif perf < 55.0 and actual_val > 15_000_000:
            multiplier = random.uniform(0.65, 0.88)  # Overvalued star
        else:
            multiplier = random.uniform(0.90, 1.15)

        pred_val = round(actual_val * multiplier)
        delta = pred_val - actual_val

        p_record = {
            "id": player_id,
            "name": full_name,
            "age": age,
            "nationality": nat,
            "club": club,
            "league": league,
            "primary_position": pos,
            "secondary_position": random.choice([None, "CM" if pos == "CAM" else "W" if pos == "ST" else None]),
            "market_value_eur": actual_val,
            "contract_expires_year": contract,
            "minutes_played": minutes,
            "matches_played": matches,
            "performance_score": perf,
            "stats": stats,
            "embedding": emb,
            "valuation": {
                "predicted_value_eur": pred_val,
                "actual_value_eur": round(actual_val),
                "undervaluation_delta_eur": round(delta),
                "undervaluation_ratio": round(pred_val / (actual_val + 1), 3),
                "confidence_score": round(random.uniform(0.80, 0.94), 2),
                "shap_drivers": [
                    {"feature": f"perf_{pos}", "impact_eur": round(abs(delta) * 0.5), "direction": "positive" if delta > 0 else "negative"},
                    {"feature": "age_factor", "impact_eur": round(abs(delta) * 0.25), "direction": "positive" if age <= 24 else "negative"},
                    {"feature": "league_coeff", "impact_eur": round(abs(delta) * 0.25), "direction": "positive" if LEAGUES_CONFIG[league]['tier'] == 1 else "negative"},
                ],
            },
        }
        players.append(p_record)
        player_id += 1

    return players


def save_dataset_json(players: List[Dict[str, Any]], filepath: str):
    """Saves generated dataset to disk for zero-dependency inspection and tests."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(players, f, indent=2, ensure_ascii=False)
    print(f"[Seed] Successfully exported {len(players)} players to {filepath}")


def seed_database_from_records(players: List[Dict[str, Any]]):
    """Inserts dataset records into live PostgreSQL with pgvector."""
    from db.connection import SyncSessionLocal, sync_engine
    from db.models import Base, Player, PlayerSeasonStats, PlayerEmbedding, PlayerValuation
    from sqlalchemy import text

    print(f"[Seed] Initializing database tables and vector extensions on {sync_engine.url}...")
    with sync_engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()

    Base.metadata.create_all(bind=sync_engine)

    session = SyncSessionLocal()
    try:
        # Clear existing data
        session.query(PlayerValuation).delete()
        session.query(PlayerEmbedding).delete()
        session.query(PlayerSeasonStats).delete()
        session.query(Player).delete()
        session.commit()

        print(f"[Seed] Seeding {len(players)} player records into PostgreSQL...")
        for p in players:
            player_obj = Player(
                id=p["id"],
                name=p["name"],
                age=p["age"],
                nationality=p["nationality"],
                club=p["club"],
                league=p["league"],
                primary_position=p["primary_position"],
                secondary_position=p["secondary_position"],
                market_value_eur=p["market_value_eur"],
                contract_expires_year=p["contract_expires_year"],
                minutes_played=p["minutes_played"],
                matches_played=p["matches_played"],
            )
            session.add(player_obj)
            session.flush()

            stats_data = p["stats"]
            stats_obj = PlayerSeasonStats(
                player_id=player_obj.id,
                **stats_data
            )
            session.add(stats_obj)

            emb_obj = PlayerEmbedding(
                player_id=player_obj.id,
                tactical_role=f"{p['primary_position']}_{p['club']}",
                embedding=p["embedding"]
            )
            session.add(emb_obj)

            val_data = p["valuation"]
            val_obj = PlayerValuation(
                player_id=player_obj.id,
                predicted_value_eur=val_data["predicted_value_eur"],
                actual_value_eur=val_data["actual_value_eur"],
                undervaluation_delta_eur=val_data["undervaluation_delta_eur"],
                undervaluation_ratio=val_data["undervaluation_ratio"],
                confidence_score=val_data["confidence_score"],
                shap_drivers_json=val_data["shap_drivers"],
            )
            session.add(val_obj)

        session.commit()
        print(f"[Seed] Seed completed successfully: {len(players)} players saved.")
    except Exception as e:
        session.rollback()
        print(f"[Seed ERROR] Failed to seed database: {e}")
        raise
    finally:
        session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed autonomous Moneyball 2.0 dataset")
    parser.add_argument("--count", type=int, default=550, help="Number of players to generate")
    parser.add_argument("--json-only", action="store_true", help="Only generate and save mock_players.json")
    parser.add_argument("--db", action="store_true", help="Seed records directly into PostgreSQL database")
    args = parser.parse_args()

    output_path = os.path.join(os.path.dirname(__file__), "mock_players.json")
    dataset = generate_full_dataset(target_count=args.count)
    save_dataset_json(dataset, output_path)

    if args.db:
        seed_database_from_records(dataset)
