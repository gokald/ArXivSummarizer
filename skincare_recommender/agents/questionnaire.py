from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Dict, Any


def load_questionnaire(path: str | Path) -> pd.DataFrame:
    """Load diagnostic questionnaire from an Excel file."""
    df = pd.read_excel(path)
    return df


def collect_answers(df: pd.DataFrame) -> Dict[str, Any]:
    """Simple CLI questionnaire returning answers as a dictionary."""
    answers: Dict[str, Any] = {}
    for _, row in df.iterrows():
        question = row.get("question") or str(row.get("Question"))
        if not isinstance(question, str):
            continue
        options = row.get("options") or row.get("Options")
        prompt = f"{question}"
        if isinstance(options, str):
            prompt += f" ({options})"
        prompt += ": "
        ans = input(prompt)
        answers[question] = ans
    return answers
