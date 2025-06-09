from __future__ import annotations

import json
from typing import List, Dict, Any


def mmr_ranking(products: List[Dict[str, Any]], scores: List[float], lambda_param: float = 0.5, top_k: int = 5) -> List[Dict[str, Any]]:
    """Perform Maximal Marginal Relevance ranking."""
    ranked = []
    selected_indices = []
    scores = list(scores)

    while len(ranked) < min(top_k, len(products)):
        best_idx = None
        best_val = -float("inf")
        for i, s in enumerate(scores):
            if i in selected_indices:
                continue
            diversity = 0.0
            for j in selected_indices:
                diversity = max(diversity, 1 - abs(scores[i] - scores[j]))
            value = lambda_param * s - (1 - lambda_param) * diversity
            if value > best_val:
                best_val = value
                best_idx = i
        if best_idx is None:
            break
        selected_indices.append(best_idx)
        ranked.append(products[best_idx] | {"score": scores[best_idx]})
    return ranked


def save_ranking(ranking: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ranking, f, indent=2)
