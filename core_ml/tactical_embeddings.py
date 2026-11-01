import os
import math
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

class TacticalEmbeddingEngine:
    """
    Tactical Style Vector Embedding Engine (64-dimensional L2-normalized).
    Provides rapid cosine similarity analog search matching pgvector HNSW behavior.
    """

    def __init__(self, vector_dim: int = 64):
        self.vector_dim = vector_dim
        self.player_ids: List[int] = []
        self.players_meta: Dict[int, Dict[str, Any]] = {}
        self.embedding_matrix: Optional[np.ndarray] = None  # Shape: (N, 64)

    @staticmethod
    def compute_vector(position: str, stats: Dict[str, float]) -> np.ndarray:
        """
        Builds a deterministic 64-dimensional unit-length tactical style embedding.
        """
        raw_features = [
            stats.get("npxg_p90", 0.0) / 1.0,
            stats.get("xag_p90", 0.0) / 0.6,
            stats.get("shots_total_p90", 0.0) / 5.0,
            stats.get("shots_on_target_pct", 0.0) / 100.0,
            stats.get("shot_creating_actions_p90", 0.0) / 8.0,
            stats.get("goal_creating_actions_p90", 0.0) / 1.2,
            stats.get("key_passes_p90", 0.0) / 4.0,
            stats.get("pass_completion_pct", 0.0) / 100.0,
            stats.get("passes_attempted_p90", 0.0) / 100.0,
            stats.get("prgp_p90", 0.0) / 10.0,
            stats.get("passes_into_final_third_p90", 0.0) / 8.0,
            stats.get("passes_into_penalty_area_p90", 0.0) / 4.0,
            stats.get("prgc_p90", 0.0) / 8.0,
            stats.get("dribbles_completed_p90", 0.0) / 5.0,
            stats.get("dribble_success_pct", 0.0) / 100.0,
            stats.get("carries_into_final_third_p90", 0.0) / 5.0,
            stats.get("dispossessed_p90", 0.0) / 3.0,
            stats.get("tackles_p90", 0.0) / 5.0,
            stats.get("tackles_won_p90", 0.0) / 3.5,
            stats.get("interceptions_p90", 0.0) / 3.0,
            stats.get("blocks_p90", 0.0) / 2.5,
            stats.get("clearances_p90", 0.0) / 8.0,
            stats.get("recoveries_p90", 0.0) / 11.0,
            stats.get("tackles_def_3rd_p90", 0.0) / 3.0,
            stats.get("tackles_mid_3rd_p90", 0.0) / 3.0,
            stats.get("tackles_att_3rd_p90", 0.0) / 1.0,
            stats.get("aerial_duels_won_pct", 0.0) / 100.0,
            stats.get("aerial_duels_total_p90", 0.0) / 7.0,
            stats.get("goals_p90", 0.0) / 1.0,
            stats.get("assists_p90", 0.0) / 0.6,
        ]

        pos_map = {"CB": 0, "FB": 1, "CM": 2, "CAM": 3, "W": 4, "ST": 5}
        pos_idx = pos_map.get(position, 2)
        pos_encoding = [0.0] * 6
        pos_encoding[pos_idx] = 1.0

        combined = raw_features + pos_encoding  # 36 dims
        extra_dims = [math.sin((i + 1) * sum(combined[:10])) * 0.5 for i in range(28)]  # 28 dims -> 64
        full = np.array(combined + extra_dims, dtype=np.float32)

        norm = np.linalg.norm(full)
        if norm > 0:
            full = full / norm
        return full

    def fit_from_dataset(self, players: List[Dict[str, Any]]):
        """Indexes all players and builds the continuous embedding matrix."""
        self.player_ids = []
        self.players_meta = {}
        matrix_rows = []

        for p in players:
            pid = p["id"]
            pos = p["primary_position"]
            stats = p.get("stats", {})

            if "embedding" in p and len(p["embedding"]) == 64:
                vec = np.array(p["embedding"], dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
            else:
                vec = self.compute_vector(pos, stats)

            self.player_ids.append(pid)
            self.players_meta[pid] = {
                "id": pid,
                "name": p["name"],
                "club": p["club"],
                "league": p["league"],
                "primary_position": pos,
                "age": p["age"],
                "market_value_eur": p["market_value_eur"],
                "valuation": p.get("valuation", {}),
                "stats": stats,
            }
            matrix_rows.append(vec)

        self.embedding_matrix = np.vstack(matrix_rows)  # (N, 64)

    def find_analogs(
        self,
        target_player_id: int,
        top_k: int = 5,
        same_position: bool = True,
        max_market_value: Optional[float] = None,
        only_undervalued: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Finds top-K tactical clones using vector cosine similarity.
        """
        if self.embedding_matrix is None or target_player_id not in self.players_meta:
            raise ValueError(f"Player ID {target_player_id} not found in embedding engine index.")

        target_idx = self.player_ids.index(target_player_id)
        target_vec = self.embedding_matrix[target_idx]  # shape (64,)
        target_meta = self.players_meta[target_player_id]

        # Cosine similarity is dot product because vectors are L2-normalized
        similarities = np.dot(self.embedding_matrix, target_vec)

        candidates = []
        for i, pid in enumerate(self.player_ids):
            if pid == target_player_id:
                continue
            meta = self.players_meta[pid]

            if same_position and meta["primary_position"] != target_meta["primary_position"]:
                continue

            if max_market_value is not None and meta["market_value_eur"] > max_market_value:
                continue

            delta = meta.get("valuation", {}).get("undervaluation_delta_eur", 0.0)
            if only_undervalued and delta <= 0:
                continue

            sim_score = float(similarities[i])
            sim_pct = round(max(0.0, min(100.0, (sim_score + 1.0) / 2.0 * 100.0)), 1)
            cosine_dist = round(float(1.0 - sim_score), 4)

            candidates.append({
                "id": meta["id"],
                "name": meta["name"],
                "club": meta["club"],
                "league": meta["league"],
                "primary_position": meta["primary_position"],
                "age": meta["age"],
                "market_value_eur": meta["market_value_eur"],
                "predicted_value_eur": meta.get("valuation", {}).get("predicted_value_eur", meta["market_value_eur"]),
                "undervaluation_delta_eur": delta,
                "similarity_score_pct": sim_pct,
                "cosine_distance": cosine_dist,
            })

        # Sort by similarity score descending
        candidates.sort(key=lambda x: x["similarity_score_pct"], reverse=True)
        return candidates[:top_k]
