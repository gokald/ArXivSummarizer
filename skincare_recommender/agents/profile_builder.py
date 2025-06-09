from __future__ import annotations

import json
from typing import Dict, Any

import pandas as pd


# Example weights from Algorithm 1 (Eq. 3-7)
WEIGHTS = {
    "dry": 1.0,
    "oily": 0.8,
    "sensitive": 1.2,
    "acne": 1.5,
}


def build_profile(answers: Dict[str, Any]) -> Dict[str, Any]:
    """Build a user skin profile applying weights defined in the algorithm."""
    profile: Dict[str, Any] = {}
    for key, val in answers.items():
        weight = WEIGHTS.get(key.lower(), 1.0)
        profile[key] = {"answer": val, "weight": weight}
    return profile


def save_profile(profile: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
