-- =============================================================================
-- Football Moneyball 2.0 & Scouting Decision Intelligence
-- PostgreSQL 18 + pgvector Initial Schema & HNSW Vector Indexing
-- =============================================================================

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Drop tables if exist in reverse order
DROP TABLE IF EXISTS scouting_dossiers CASCADE;
DROP TABLE IF EXISTS player_valuations CASCADE;
DROP TABLE IF EXISTS player_embeddings CASCADE;
DROP TABLE IF EXISTS player_season_stats CASCADE;
DROP TABLE IF EXISTS players CASCADE;

-- 1. Core Players Table
CREATE TABLE players (
    id SERIAL PRIMARY KEY,
    name VARCHAR(150) NOT NULL,
    age INT NOT NULL,
    nationality VARCHAR(80) NOT NULL,
    club VARCHAR(120) NOT NULL,
    league VARCHAR(80) NOT NULL,
    primary_position VARCHAR(10) NOT NULL,  -- CB, FB, CM, CAM, W, ST
    secondary_position VARCHAR(10),
    market_value_eur DOUBLE PRECISION NOT NULL,
    contract_expires_year INT NOT NULL,
    minutes_played INT NOT NULL DEFAULT 0,
    matches_played INT NOT NULL DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_players_position ON players (primary_position);
CREATE INDEX idx_players_league ON players (league);
CREATE INDEX idx_players_market_value ON players (market_value_eur);

-- 2. FBref / StatsBomb Standardized Performance Metrics (p90)
CREATE TABLE player_season_stats (
    id SERIAL PRIMARY KEY,
    player_id INT NOT NULL UNIQUE REFERENCES players(id) ON DELETE CASCADE,
    season VARCHAR(15) NOT NULL DEFAULT '2025/2026',
    
    -- Finishing & Direct Threat
    goals_p90 DOUBLE PRECISION DEFAULT 0.0,
    assists_p90 DOUBLE PRECISION DEFAULT 0.0,
    npxg_p90 DOUBLE PRECISION DEFAULT 0.0,
    shots_total_p90 DOUBLE PRECISION DEFAULT 0.0,
    shots_on_target_pct DOUBLE PRECISION DEFAULT 0.0,

    -- Playmaking & Chance Creation
    xag_p90 DOUBLE PRECISION DEFAULT 0.0,
    shot_creating_actions_p90 DOUBLE PRECISION DEFAULT 0.0,
    goal_creating_actions_p90 DOUBLE PRECISION DEFAULT 0.0,
    key_passes_p90 DOUBLE PRECISION DEFAULT 0.0,

    -- Passing & Distribution
    pass_completion_pct DOUBLE PRECISION DEFAULT 0.0,
    passes_attempted_p90 DOUBLE PRECISION DEFAULT 0.0,
    prgp_p90 DOUBLE PRECISION DEFAULT 0.0,  -- Progressive passes
    passes_into_final_third_p90 DOUBLE PRECISION DEFAULT 0.0,
    passes_into_penalty_area_p90 DOUBLE PRECISION DEFAULT 0.0,

    -- Ball Carrying & Take-ons
    prgc_p90 DOUBLE PRECISION DEFAULT 0.0,  -- Progressive carries
    dribbles_completed_p90 DOUBLE PRECISION DEFAULT 0.0,
    dribble_success_pct DOUBLE PRECISION DEFAULT 0.0,
    carries_into_final_third_p90 DOUBLE PRECISION DEFAULT 0.0,
    dispossessed_p90 DOUBLE PRECISION DEFAULT 0.0,

    -- Defensive Actions & Pressure
    tackles_p90 DOUBLE PRECISION DEFAULT 0.0,
    tackles_won_p90 DOUBLE PRECISION DEFAULT 0.0,
    interceptions_p90 DOUBLE PRECISION DEFAULT 0.0,
    blocks_p90 DOUBLE PRECISION DEFAULT 0.0,
    clearances_p90 DOUBLE PRECISION DEFAULT 0.0,
    recoveries_p90 DOUBLE PRECISION DEFAULT 0.0,
    tackles_def_3rd_p90 DOUBLE PRECISION DEFAULT 0.0,
    tackles_mid_3rd_p90 DOUBLE PRECISION DEFAULT 0.0,
    tackles_att_3rd_p90 DOUBLE PRECISION DEFAULT 0.0,

    -- Aerial & Physical
    aerial_duels_won_pct DOUBLE PRECISION DEFAULT 0.0,
    aerial_duels_total_p90 DOUBLE PRECISION DEFAULT 0.0
);

-- 3. Tactical Style Vector Embeddings (pgvector 64-d with HNSW Index)
CREATE TABLE player_embeddings (
    id SERIAL PRIMARY KEY,
    player_id INT NOT NULL UNIQUE REFERENCES players(id) ON DELETE CASCADE,
    tactical_role VARCHAR(50) NOT NULL,
    embedding vector(64) NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- HNSW Vector Index for Lightning-Fast Cosine Similarity Search (< 10ms)
CREATE INDEX idx_player_embeddings_hnsw 
ON player_embeddings 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 4. ML Market Valuation, Delta & SHAP Drivers
CREATE TABLE player_valuations (
    id SERIAL PRIMARY KEY,
    player_id INT NOT NULL UNIQUE REFERENCES players(id) ON DELETE CASCADE,
    predicted_value_eur DOUBLE PRECISION NOT NULL,
    actual_value_eur DOUBLE PRECISION NOT NULL,
    undervaluation_delta_eur DOUBLE PRECISION NOT NULL,  -- predicted - actual
    undervaluation_ratio DOUBLE PRECISION NOT NULL,      -- predicted / (actual + 1)
    confidence_score DOUBLE PRECISION NOT NULL DEFAULT 0.85,
    shap_drivers_json JSONB,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_valuations_delta ON player_valuations (undervaluation_delta_eur DESC);

-- 5. AI Transfer Audit Dossiers (Gemini 3.8 Flash Reports)
CREATE TABLE scouting_dossiers (
    id SERIAL PRIMARY KEY,
    target_player_id INT NOT NULL REFERENCES players(id) ON DELETE CASCADE,
    analog_player_id INT NOT NULL REFERENCES players(id) ON DELETE CASCADE,
    tactical_similarity_score DOUBLE PRECISION NOT NULL,
    executive_summary TEXT NOT NULL,
    financial_arbitrage_json JSONB,
    strengths_json JSONB,
    risks_json JSONB,
    verdict VARCHAR(30) NOT NULL,
    raw_llm_markdown TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_target_analog UNIQUE (target_player_id, analog_player_id)
);

