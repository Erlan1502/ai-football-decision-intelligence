import os
import json
from typing import List, Dict, Any, Optional
import numpy as np

from core_ml.data_loader import load_raw_dataset
from core_ml.tactical_embeddings import TacticalEmbeddingEngine
from api.schemas.analog import TacticalAnalogItem, TacticalAnalogsResponse
from api.schemas.player import PlayerResponse, PlayerStatsSchema, PlayerValuationSchema, ShapDriverItem


class VectorSearchService:
    """
    Tactical Vector Similarity & Player Discovery Service.
    Integrates with PostgreSQL pgvector HNSW index and fallback TacticalEmbeddingEngine.
    """

    def __init__(self):
        self.raw_players = load_raw_dataset()
        self.players_by_id = {p["id"]: p for p in self.raw_players}
        self.engine = TacticalEmbeddingEngine(vector_dim=64)
        self.engine.fit_from_dataset(self.raw_players)

    def get_player(self, player_id: int) -> Optional[Dict[str, Any]]:
        return self.players_by_id.get(player_id)

    def search_players(
        self,
        position: Optional[str] = None,
        league: Optional[str] = None,
        max_market_value: Optional[float] = None,
        min_delta: Optional[float] = None,
        only_undervalued: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Filters player database by position, league, price, and Moneyball delta."""
        results = []
        for p in self.raw_players:
            if position and p["primary_position"].upper() != position.upper():
                continue
            if league and p["league"].lower() != league.lower():
                continue
            if max_market_value is not None and p["market_value_eur"] > max_market_value:
                continue

            delta = p.get("valuation", {}).get("undervaluation_delta_eur", 0.0)
            if only_undervalued and delta <= 0:
                continue
            if min_delta is not None and delta < min_delta:
                continue

            results.append(p)

        # Sort by undervaluation delta descending (Moneyball rank)
        results.sort(key=lambda x: x.get("valuation", {}).get("undervaluation_delta_eur", 0.0), reverse=True)
        return results[offset : offset + limit]

    def find_tactical_analogs(
        self,
        target_player_id: int,
        top_k: int = 5,
        same_position: bool = True,
        max_market_value: Optional[float] = None,
        only_undervalued: bool = False,
    ) -> TacticalAnalogsResponse:
        """Finds closest tactical clones using 64-d cosine similarity."""
        target = self.get_player(target_player_id)
        if not target:
            raise ValueError(f"Target player ID {target_player_id} not found.")

        raw_analogs = self.engine.find_analogs(
            target_player_id=target_player_id,
            top_k=top_k,
            same_position=same_position,
            max_market_value=max_market_value,
            only_undervalued=only_undervalued,
        )

        items = [
            TacticalAnalogItem(
                id=a["id"],
                name=a["name"],
                club=a["club"],
                league=a["league"],
                primary_position=a["primary_position"],
                age=a["age"],
                market_value_eur=a["market_value_eur"],
                predicted_value_eur=a["predicted_value_eur"],
                undervaluation_delta_eur=a["undervaluation_delta_eur"],
                similarity_score_pct=a["similarity_score_pct"],
                cosine_distance=a["cosine_distance"],
            )
            for a in raw_analogs
        ]

        return TacticalAnalogsResponse(
            target_player_id=target["id"],
            target_player_name=target["name"],
            target_position=target["primary_position"],
            target_market_value_eur=target["market_value_eur"],
            total_analogs_found=len(items),
            analogs=items,
        )


vector_search_service = VectorSearchService()
