from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TypedDict, Dict, Any, List

from langgraph.graph import StateGraph, END, START
from rich.console import Console

from . import (
    questionnaire,
    profile_builder,
    product_filter,
    similarity_scorer,
    rank_aggregator,
    explanation_llm,
)

console = Console()


class RecommenderState(TypedDict, total=False):
    answers: Dict[str, Any]
    profile: Dict[str, Any]
    filtered: List[Dict[str, Any]]
    scores: List[float]
    ranking: List[Dict[str, Any]]
    explanation: str


async def node_questionnaire(state: RecommenderState, question_path: str, output_dir: Path) -> RecommenderState:
    df = questionnaire.load_questionnaire(question_path)
    answers = questionnaire.collect_answers(df)
    (output_dir / "answers.json").write_text(json.dumps(answers, indent=2))
    return {"answers": answers}


def node_profile(state: RecommenderState, output_dir: Path) -> RecommenderState:
    profile = profile_builder.build_profile(state["answers"])
    profile_builder.save_profile(profile, output_dir / "profile.json")
    return {"profile": profile}


def node_filter(state: RecommenderState, catalog_path: str, output_dir: Path) -> RecommenderState:
    products = product_filter.load_catalog(catalog_path)
    filtered = product_filter.filter_products(products, state["profile"])
    product_filter.save_products(filtered, output_dir / "filtered.json")
    return {"filtered": filtered}


def node_similarity(state: RecommenderState, output_dir: Path) -> RecommenderState:
    scores = similarity_scorer.score_similarity(state["filtered"], state["profile"])
    similarity_scorer.save_scores(scores, output_dir / "scores.json")
    return {"scores": scores}


def node_rank(state: RecommenderState, output_dir: Path) -> RecommenderState:
    ranking = rank_aggregator.mmr_ranking(state["filtered"], state["scores"])
    rank_aggregator.save_ranking(ranking, output_dir / "ranking.json")
    return {"ranking": ranking}


async def node_explanation(state: RecommenderState, api_key: str, output_dir: Path) -> RecommenderState:
    explanation = await explanation_llm.generate_explanation(api_key, state["ranking"])
    explanation_llm.save_explanation(explanation, output_dir / "explanation.json")
    return {"explanation": explanation}


async def run_pipeline(questionnaire_path: str, catalog_path: str, api_key: str, output_dir: str = "outputs") -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    graph = StateGraph(RecommenderState)

    async def q_node(state: RecommenderState) -> RecommenderState:
        return await node_questionnaire(state, questionnaire_path, out_dir)

    def profile_node(state: RecommenderState) -> RecommenderState:
        return node_profile(state, out_dir)

    def filter_node(state: RecommenderState) -> RecommenderState:
        return node_filter(state, catalog_path, out_dir)

    def similarity_node(state: RecommenderState) -> RecommenderState:
        return node_similarity(state, out_dir)

    def rank_node(state: RecommenderState) -> RecommenderState:
        return node_rank(state, out_dir)

    async def explain_node(state: RecommenderState) -> RecommenderState:
        return await node_explanation(state, api_key, out_dir)

    graph.add_node("ask_questions", q_node)
    graph.add_node("build_profile", profile_node)
    graph.add_node("filter_products", filter_node)
    graph.add_node("score_similarity", similarity_node)
    graph.add_node("rank_products", rank_node)
    graph.add_node("explain", explain_node)

    graph.add_edge(START, "ask_questions")
    graph.add_edge("ask_questions", "build_profile")
    graph.add_edge("build_profile", "filter_products")
    graph.add_edge("filter_products", "score_similarity")
    graph.add_edge("score_similarity", "rank_products")
    graph.add_edge("rank_products", "explain")
    graph.add_edge("explain", END)

    compiled = graph.compile()

    result: RecommenderState = await compiled.ainvoke({})
    # final explanation
    console.print("\n[bold underline]Recommendations[/bold underline]")
    for item in result.get("ranking", []):
        console.print(f"- {item.get('name')} (score {item.get('score'):.2f})")
    console.print("\n[bold]Explanation:[/bold]", result.get("explanation", ""))


async def main(questionnaire_path: str, catalog_path: str, api_key: str, output_dir: str = "outputs") -> None:
    await run_pipeline(questionnaire_path, catalog_path, api_key, output_dir)


if __name__ == "__main__":
    import os
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m skincare_recommender.agents.orchestrator QUESTIONNAIRE.xlsx CATALOG.json")
        sys.exit(1)

    q_path = sys.argv[1]
    cat_path = sys.argv[2]
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY environment variable required")
        sys.exit(1)
    asyncio.run(main(q_path, cat_path, api_key))
