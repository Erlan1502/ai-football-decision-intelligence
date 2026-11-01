import time
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds

FORMATIONS = {
    "4-3-3": {"CB": 3, "FB": 2, "CM": 2, "CAM": 1, "W": 2, "ST": 1},  # 3 CB/anchors + 2 FB + 3 MID + 3 FWD = 11
    "4-2-3-1": {"CB": 3, "FB": 2, "CM": 2, "CAM": 1, "W": 2, "ST": 1},
    "3-5-2": {"CB": 3, "FB": 2, "CM": 2, "CAM": 1, "W": 1, "ST": 2},  # 3 CB + 2 FB + 3 MID + 1 W + 2 ST = 11
}


class SquadOptimizer:
    """
    Integer Linear Programming (MILP) Squad Builder using Scipy Optimize.
    Maximizes cumulative Moneyball performance score subject to a budget ceiling,
    tactical formation quotas, and club limits.
    """

    def __init__(self, players_data: Optional[List[Dict[str, Any]]] = None):
        self.players_df = None
        if players_data:
            self.set_players(players_data)

    def set_players(self, players: List[Dict[str, Any]]):
        rows = []
        for p in players:
            val = p.get("valuation", {})
            rows.append({
                "id": p["id"],
                "name": p["name"],
                "club": p["club"],
                "league": p["league"],
                "primary_position": p["primary_position"],
                "age": p["age"],
                "market_value_eur": float(p["market_value_eur"]),
                "performance_score": float(p.get("performance_score", 65.0)),
                "predicted_value_eur": float(val.get("predicted_value_eur", p["market_value_eur"])),
                "undervaluation_delta_eur": float(val.get("undervaluation_delta_eur", 0.0)),
            })
        self.players_df = pd.DataFrame(rows)

    def optimize_squad(
        self,
        budget_cap_eur: float = 120_000_000.0,
        formation: str = "4-3-3",
        max_per_club: int = 3,
        moneyball_weight: float = 0.35,
        min_delta_filter: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Solves the 0-1 Knapsack MILP problem for 11 players.
        
        Args:
            budget_cap_eur: Transfer budget ceiling in EUR.
            formation: "4-3-3", "4-2-3-1", or "3-5-2".
            max_per_club: Max allowed players from the same club.
            moneyball_weight: Weight given to undervaluation delta (vs raw performance).
            min_delta_filter: Filter pool to only players with delta >= threshold.
        """
        t0 = time.time()
        if self.players_df is None or len(self.players_df) < 11:
            raise ValueError("Insufficient players in pool for optimization.")

        df = self.players_df.copy()
        if min_delta_filter is not None:
            filtered = df[df["undervaluation_delta_eur"] >= min_delta_filter]
            if len(filtered) >= 15:
                df = filtered.reset_index(drop=True)

        n = len(df)
        pos_quotas = FORMATIONS.get(formation, FORMATIONS["4-3-3"])

        # Objective Function: Maximize (c^T x) -> Minimize (-c^T x)
        # Score combines performance score and normalized undervaluation delta
        perf_norm = (df["performance_score"] - 40.0) / 20.0
        delta_norm = df["undervaluation_delta_eur"] / 10_000_000.0
        objective_scores = (1.0 - moneyball_weight) * perf_norm + moneyball_weight * delta_norm
        c = -objective_scores.values  # Minimize negative score

        # Binary decision variables x_i in {0, 1}
        integrality = np.ones(n)  # 1 = integer variable
        bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))

        constraints_A = []
        lhs = []
        rhs = []

        # 1. Total players == 11
        constraints_A.append(np.ones(n))
        lhs.append(11.0)
        rhs.append(11.0)

        # 2. Total cost <= budget_cap_eur
        constraints_A.append(df["market_value_eur"].values)
        lhs.append(0.0)
        rhs.append(float(budget_cap_eur))

        # 3. Positional exact equality constraints
        for pos, quota in pos_quotas.items():
            pos_mask = (df["primary_position"] == pos).astype(float).values
            constraints_A.append(pos_mask)
            lhs.append(float(quota))
            rhs.append(float(quota))

        # 4. Max players per club constraint
        clubs = df["club"].unique()
        for club in clubs:
            club_mask = (df["club"] == club).astype(float).values
            constraints_A.append(club_mask)
            lhs.append(0.0)
            rhs.append(float(max_per_club))

        A_mat = np.array(constraints_A)
        linear_constraint = LinearConstraint(A_mat, lhs, rhs)

        # Solve MILP
        res = milp(c=c, integrality=integrality, bounds=bounds, constraints=linear_constraint)

        elapsed = round(time.time() - t0, 3)
        if not res.success:
            # Fallback if budget was too restrictive: return best effort or error
            return {
                "success": False,
                "message": f"Optimization infeasible with budget €{budget_cap_eur:,.0f} and formation {formation}: {res.status}",
                "solve_time_sec": elapsed,
                "squad": [],
            }

        chosen_indices = np.where(res.x > 0.5)[0]
        selected_df = df.iloc[chosen_indices].copy()

        total_spent = float(selected_df["market_value_eur"].sum())
        total_delta = float(selected_df["undervaluation_delta_eur"].sum())
        avg_perf = float(selected_df["performance_score"].mean())

        squad_list = selected_df.to_dict(orient="records")

        return {
            "success": True,
            "formation": formation,
            "budget_cap_eur": float(budget_cap_eur),
            "total_spent_eur": total_spent,
            "total_cost_eur": total_spent,
            "budget_remaining_eur": round(budget_cap_eur - total_spent, 2),
            "total_undervaluation_delta_eur": round(total_delta, 2),
            "avg_performance_score": round(avg_perf, 2),
            "total_performance_score": round(float(selected_df["performance_score"].sum()), 2),
            "total_players": len(squad_list),
            "solve_time_sec": elapsed,
            "solve_time_seconds": elapsed,
            "squad": squad_list,
            "selected_players": squad_list,
        }

