from __future__ import annotations

import json
from typing import List, Dict, Any

from openai import AsyncOpenAI


async def generate_explanation(api_key: str, ranking: List[Dict[str, Any]]) -> str:
    """Generate explanations for the ranked products using OpenAI asynchronously."""
    client = AsyncOpenAI(api_key=api_key)
    prompt = "Provide a concise explanation for recommending the following products:\n"
    for item in ranking:
        prompt += f"- {item.get('name')} (score {item.get('score'):.2f})\n"
    resp = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
    )
    explanation = resp.choices[0].message.content
    return explanation


def save_explanation(text: str, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"explanation": text}, f, indent=2)
