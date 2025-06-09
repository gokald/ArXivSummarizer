from __future__ import annotations

import os
import asyncio
from pathlib import Path
from rich.console import Console

from .agents import orchestrator

console = Console()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Skin-care product recommender")
    parser.add_argument("questionnaire", help="Path to diagnostic questionnaire Excel file")
    parser.add_argument("catalog", help="Path to product catalog JSON file")
    parser.add_argument("--output", default="outputs", help="Directory to save outputs")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]OPENAI_API_KEY environment variable required[/red]")
        raise SystemExit(1)

    asyncio.run(orchestrator.main(args.questionnaire, args.catalog, api_key, args.output))


if __name__ == "__main__":
    main()
