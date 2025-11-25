"""
Reflector: Evaluates recommendation quality and suggests improvements.
"""

import logging
from typing import Dict, List, Any
from dataclasses import dataclass
from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class Reflection:
    """Result of reflecting on recommendations."""
    success: bool
    quality_score: float  # 0.0-1.0
    issues: List[str]
    suggestions: List[str]
    should_retry: bool
    adjusted_parameters: Dict[str, Any]


class Reflector:
    """Evaluates recommendation quality and suggests improvements."""

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def evaluate(self, query: str, results: List[Dict], plan_step: Any) -> Reflection:
        """
        Evaluate if results meet the query intent.

        Args:
            query: Original user query
            results: Recommendations returned
            plan_step: The step that was executed

        Returns:
            Reflection with quality assessment
        """
        logger.info(f"Reflecting on {len(results)} results for: {query}")

        if len(results) == 0:
            return Reflection(
                success=False,
                quality_score=0.0,
                issues=["No results returned"],
                suggestions=["Try relaxing constraints", "Use broader search terms"],
                should_retry=True,
                adjusted_parameters={}
            )

        # Simple heuristic-based evaluation
        quality_score = min(1.0, len(results) / 5.0)  # Target 5+ results

        issues = []
        suggestions = []

        if len(results) < 3:
            issues.append("Too few results")
            suggestions.append("Relax constraints or broaden search")

        return Reflection(
            success=quality_score >= 0.6,
            quality_score=quality_score,
            issues=issues,
            suggestions=suggestions,
            should_retry=quality_score < 0.4,
            adjusted_parameters={}
        )
