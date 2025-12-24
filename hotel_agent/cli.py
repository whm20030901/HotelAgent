from __future__ import annotations

import argparse
from pathlib import Path

from .agent import SentimentAgent
from .model import ModelTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hotel review sentiment agent")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models"),
        help="Directory to save/load the trained model",
    )
    parser.add_argument(
        "--train",
        type=Path,
        help="CSV dataset path with label/review columns",
    )
    parser.add_argument(
        "--analyze",
        type=str,
        help="Review text to analyze",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    model_path = args.model_dir / "sentiment_model.joblib"

    if args.train:
        trainer = ModelTrainer()
        pipeline = trainer.train(args.train)
        model_path = trainer.save(pipeline, args.model_dir)
        print(f"Model saved to {model_path}")

    if args.analyze:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run with --train first."
            )
        agent = SentimentAgent(model_path)
        result = agent.run(args.analyze)
        print(result["summary"])


if __name__ == "__main__":
    main()
