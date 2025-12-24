from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from langchain_core.tools import tool
from langgraph.graph import StateGraph

from .model import SentimentAnalyzer


class ReviewState(TypedDict, total=False):
    review: str
    label: int
    confidence: float
    summary: str


@dataclass
class SentimentAgent:
    model_path: Path

    def _build_tool(self):
        analyzer = SentimentAnalyzer(self.model_path)

        @tool("analyze_review")
        def analyze_review(review: str) -> dict:
            """Analyze hotel review sentiment; returns label and confidence."""
            label, confidence = analyzer.predict(review)
            sentiment = "好评" if label == 1 else "差评"
            return {
                "label": label,
                "confidence": round(confidence, 2),
                "sentiment": sentiment,
            }

        return analyze_review

    def build_graph(self):
        analyze_review = self._build_tool()

        def analyze_node(state: ReviewState) -> ReviewState:
            result = analyze_review.invoke(state["review"])
            return {
                "review": state["review"],
                "label": int(result["label"]),
                "confidence": float(result["confidence"]),
                "summary": f"{result['sentiment']} (label={result['label']}, confidence={result['confidence']:.2f})",
            }

        graph = StateGraph(ReviewState)
        graph.add_node("analyze", analyze_node)
        graph.set_entry_point("analyze")
        graph.set_finish_point("analyze")
        return graph.compile()

    def run(self, review: str) -> ReviewState:
        app = self.build_graph()
        return app.invoke({"review": review})
