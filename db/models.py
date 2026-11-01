import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    ForeignKey,
    DateTime,
    Text,
    Index,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector

Base = declarative_base()


class Player(Base):
    __tablename__ = "players"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(150), nullable=False)
    age = Column(Integer, nullable=False)
    nationality = Column(String(80), nullable=False)
    club = Column(String(120), nullable=False)
    league = Column(String(80), nullable=False)
    primary_position = Column(String(10), nullable=False, index=True)  # CB, FB, CM, CAM, W, ST
    secondary_position = Column(String(10), nullable=True)
    market_value_eur = Column(Float, nullable=False, index=True)
    contract_expires_year = Column(Integer, nullable=False)
    minutes_played = Column(Integer, default=0, nullable=False)
    matches_played = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.datetime.utcnow)

    # Relationships
    stats = relationship("PlayerSeasonStats", back_populates="player", uselist=False, cascade="all, delete-orphan")
    embedding_rel = relationship("PlayerEmbedding", back_populates="player", uselist=False, cascade="all, delete-orphan")
    valuation = relationship("PlayerValuation", back_populates="player", uselist=False, cascade="all, delete-orphan")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "age": self.age,
            "nationality": self.nationality,
            "club": self.club,
            "league": self.league,
            "primary_position": self.primary_position,
            "secondary_position": self.secondary_position,
            "market_value_eur": self.market_value_eur,
            "contract_expires_year": self.contract_expires_year,
            "minutes_played": self.minutes_played,
            "matches_played": self.matches_played,
        }


class PlayerSeasonStats(Base):
    """
    Standardized per-90 metrics matching FBref / StatsBomb data schemas.
    """
    __tablename__ = "player_season_stats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(Integer, ForeignKey("players.id", ondelete="CASCADE"), unique=True, nullable=False)
    season = Column(String(15), default="2025/2026", nullable=False)

    # Finishing & Direct Threat
    goals_p90 = Column(Float, default=0.0)
    assists_p90 = Column(Float, default=0.0)
    npxg_p90 = Column(Float, default=0.0)
    shots_total_p90 = Column(Float, default=0.0)
    shots_on_target_pct = Column(Float, default=0.0)

    # Playmaking & Chance Creation
    xag_p90 = Column(Float, default=0.0)
    shot_creating_actions_p90 = Column(Float, default=0.0)
    goal_creating_actions_p90 = Column(Float, default=0.0)
    key_passes_p90 = Column(Float, default=0.0)

    # Passing & Distribution
    pass_completion_pct = Column(Float, default=0.0)
    passes_attempted_p90 = Column(Float, default=0.0)
    prgp_p90 = Column(Float, default=0.0)  # Progressive passes
    passes_into_final_third_p90 = Column(Float, default=0.0)
    passes_into_penalty_area_p90 = Column(Float, default=0.0)

    # Ball Carrying & Take-ons
    prgc_p90 = Column(Float, default=0.0)  # Progressive carries
    dribbles_completed_p90 = Column(Float, default=0.0)
    dribble_success_pct = Column(Float, default=0.0)
    carries_into_final_third_p90 = Column(Float, default=0.0)
    dispossessed_p90 = Column(Float, default=0.0)

    # Defensive Actions & Pressure
    tackles_p90 = Column(Float, default=0.0)
    tackles_won_p90 = Column(Float, default=0.0)
    interceptions_p90 = Column(Float, default=0.0)
    blocks_p90 = Column(Float, default=0.0)
    clearances_p90 = Column(Float, default=0.0)
    recoveries_p90 = Column(Float, default=0.0)
    tackles_def_3rd_p90 = Column(Float, default=0.0)
    tackles_mid_3rd_p90 = Column(Float, default=0.0)
    tackles_att_3rd_p90 = Column(Float, default=0.0)

    # Aerial & Physical
    aerial_duels_won_pct = Column(Float, default=0.0)
    aerial_duels_total_p90 = Column(Float, default=0.0)

    player = relationship("Player", back_populates="stats")

    def to_dict(self) -> Dict[str, Any]:
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class PlayerEmbedding(Base):
    """
    Tactical style embedding (64 dimensions) indexed via HNSW for cosine similarity search.
    """
    __tablename__ = "player_embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(Integer, ForeignKey("players.id", ondelete="CASCADE"), unique=True, nullable=False)
    tactical_role = Column(String(50), nullable=False)
    embedding = Column(Vector(64), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=datetime.datetime.utcnow)

    player = relationship("Player", back_populates="embedding_rel")

    __table_args__ = (
        Index(
            "idx_player_embeddings_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class PlayerValuation(Base):
    """
    ML estimated market value, undervaluation delta, and SHAP top drivers.
    """
    __tablename__ = "player_valuations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(Integer, ForeignKey("players.id", ondelete="CASCADE"), unique=True, nullable=False)
    predicted_value_eur = Column(Float, nullable=False)
    actual_value_eur = Column(Float, nullable=False)
    undervaluation_delta_eur = Column(Float, nullable=False, index=True)
    undervaluation_ratio = Column(Float, nullable=False)
    confidence_score = Column(Float, default=0.85, nullable=False)
    shap_drivers_json = Column(JSONB, nullable=True)
    updated_at = Column(DateTime(timezone=True), default=datetime.datetime.utcnow)

    player = relationship("Player", back_populates="valuation")


class ScoutingDossier(Base):
    """
    Structured scouting audits generated by Gemini 3.8 Flash.
    """
    __tablename__ = "scouting_dossiers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    target_player_id = Column(Integer, ForeignKey("players.id", ondelete="CASCADE"), nullable=False)
    analog_player_id = Column(Integer, ForeignKey("players.id", ondelete="CASCADE"), nullable=False)
    tactical_similarity_score = Column(Float, nullable=False)
    executive_summary = Column(Text, nullable=False)
    financial_arbitrage_json = Column(JSONB, nullable=True)
    strengths_json = Column(JSONB, nullable=True)
    risks_json = Column(JSONB, nullable=True)
    verdict = Column(String(30), nullable=False)
    raw_llm_markdown = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("target_player_id", "analog_player_id", name="uq_target_analog"),
    )

