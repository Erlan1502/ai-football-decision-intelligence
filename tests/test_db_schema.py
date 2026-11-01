import pytest
from pgvector.sqlalchemy import Vector
from db.models import (
    Base,
    Player,
    PlayerSeasonStats,
    PlayerEmbedding,
    PlayerValuation,
    ScoutingDossier,
)


def test_models_metadata_registration():
    """Verify all 5 tables are registered in SQLAlchemy Base metadata."""
    registered_tables = Base.metadata.tables.keys()
    expected = {
        "players",
        "player_season_stats",
        "player_embeddings",
        "player_valuations",
        "scouting_dossiers",
    }
    assert expected.issubset(registered_tables), f"Missing tables: {expected - registered_tables}"


def test_player_embedding_vector_dimensions():
    """Verify that PlayerEmbedding.embedding column uses Vector(64)."""
    emb_col = PlayerEmbedding.__table__.columns["embedding"]
    assert isinstance(emb_col.type, Vector), "Column must be an instance of pgvector Vector"
    assert emb_col.type.dim == 64, f"Vector dimension must be 64, got {emb_col.type.dim}"


def test_hnsw_index_configuration():
    """Verify HNSW index properties for cosine similarity."""
    indexes = PlayerEmbedding.__table__.indexes
    hnsw_idx = next((idx for idx in indexes if idx.name == "idx_player_embeddings_hnsw"), None)
    assert hnsw_idx is not None, "Index idx_player_embeddings_hnsw not found"
    assert hnsw_idx.dialect_options["postgresql"]["using"] == "hnsw"
    assert hnsw_idx.dialect_options["postgresql"]["with"] == {"m": 16, "ef_construction": 64}
    assert hnsw_idx.dialect_options["postgresql"]["ops"] == {"embedding": "vector_cosine_ops"}


def test_scouting_dossier_unique_constraint():
    """Verify unique constraint on (target_player_id, analog_player_id)."""
    table = ScoutingDossier.__table__
    uq_names = [c.name for c in table.constraints if c.name == "uq_target_analog"]
    assert len(uq_names) == 1, "Expected unique constraint 'uq_target_analog'"


def test_player_season_stats_p90_columns():
    """Verify that player_season_stats contains key FBref/StatsBomb p90 columns."""
    stats_cols = set(PlayerSeasonStats.__table__.columns.keys())
    required_p90 = {
        "npxg_p90", "xag_p90", "prgp_p90", "prgc_p90",
        "tackles_won_p90", "interceptions_p90", "recoveries_p90",
        "shot_creating_actions_p90", "aerial_duels_won_pct", "pass_completion_pct"
    }
    assert required_p90.issubset(stats_cols), f"Missing p90 columns: {required_p90 - stats_cols}"

