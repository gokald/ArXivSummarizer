from __future__ import annotations

import json
from typing import Dict, List, Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


VECTOR_FIELDS = ["ingredients", "description"]


def build_product_matrix(products: List[Dict[str, Any]]) -> tuple[np.ndarray, List[str]]:
    texts = []
    for product in products:
        text = " ".join(str(product.get(field, "")) for field in VECTOR_FIELDS)
        texts.append(text)
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    return matrix, vectorizer


def score_similarity(products: List[Dict[str, Any]], profile: Dict[str, Any]) -> List[float]:
    matrix, vectorizer = build_product_matrix(products)
    profile_text = " ".join(key for key in profile.keys())
    profile_vec = vectorizer.transform([profile_text])
    sims = cosine_similarity(profile_vec, matrix).flatten()
    return sims.tolist()


def save_scores(scores: List[float], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)
