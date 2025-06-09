from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd


def load_catalog(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("products", [])
    return data


def filter_products(products: List[Dict[str, Any]], profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Filter products based on profile labels."""
    filtered = []
    for product in products:
        tags = {t.lower() for t in product.get("tags", [])}
        match = True
        for trait in profile.keys():
            if trait.lower() in tags:
                continue
            match = False
            break
        if match:
            filtered.append(product)
    return filtered


def save_products(products: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(products, f, indent=2)
