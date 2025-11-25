"""
Tool: Constraint Relaxer

Suggests constraint relaxations when no results are found.
"""

import json
import logging
from typing import Dict, Any
from langchain.tools import StructuredTool
from pydantic.v1 import BaseModel, Field

logger = logging.getLogger(__name__)


class ConstraintRelaxerInput(BaseModel):
    """Input schema for constraint relaxer."""
    requirements: Dict[str, Any] = Field(default={}, description="Dictionary of current constraints (e.g., {'min_rating': 8.0, 'max_hours': 2.0, 'mood': 'happy'})")
    results_count: int = Field(default=0, description="Number of results found with current constraints")


class ConstraintRelaxer:
    """Suggests how to relax constraints when searches fail."""

    def __init__(self, vector_store):
        self.vector_store = vector_store

    def suggest_relaxations(self, requirements: Dict[str, Any], results_count: int) -> Dict[str, Any]:
        """
        Suggest constraint relaxations when results are insufficient.

        Args:
            requirements: Dictionary of current constraints
            results_count: Number of results found with current constraints

        Returns:
            Suggestions for relaxing constraints
        """
        logger.info(f"Generating relaxation suggestions for {len(requirements)} constraints ({results_count} results)")

        suggestions = []
        priority_order = []

        try:
            # Analyze each constraint and suggest relaxation
            if 'min_rating' in requirements:
                rating = requirements['min_rating']
                if rating >= 8.0:
                    suggestions.append({
                        "constraint": "min_rating",
                        "current": rating,
                        "suggested": max(rating - 1.0, 6.0),
                        "reason": "Lowering rating threshold will find more options",
                        "priority": 1
                    })
                    priority_order.append(1)

            if 'max_hours' in requirements:
                hours = requirements['max_hours']
                if hours < 2.5:
                    suggestions.append({
                        "constraint": "max_hours",
                        "current": hours,
                        "suggested": hours + 0.5,
                        "reason": "Allowing slightly longer runtime opens more possibilities",
                        "priority": 2
                    })
                    priority_order.append(2)

            if 'context' in requirements:
                context = requirements['context']
                relaxations = {
                    "family_watch": "Try 'solo_watch' for more mature options",
                    "date_night": "Try 'solo_watch' for broader selection",
                    "background_watch": "Try 'solo_watch' for more engaging content"
                }
                if context in relaxations:
                    suggestions.append({
                        "constraint": "context",
                        "current": context,
                        "suggested": "solo_watch",
                        "reason": relaxations[context],
                        "priority": 3
                    })
                    priority_order.append(3)

            if 'mood' in requirements:
                mood = requirements['mood']
                related_moods = {
                    "happy": ["excited", "relaxed"],
                    "sad": ["nostalgic", "inspired"],
                    "scared": ["stressed", "curious"],
                    "excited": ["happy", "bored"],
                    "curious": ["inspired", "excited"]
                }
                if mood in related_moods:
                    suggestions.append({
                        "constraint": "mood",
                        "current": mood,
                        "suggested": related_moods[mood],
                        "reason": f"Try related moods: {', '.join(related_moods[mood])}",
                        "priority": 4
                    })
                    priority_order.append(4)

            if 'type' in requirements:
                content_type = requirements['type']
                other_type = "series" if content_type == "movie" else "movie"
                suggestions.append({
                    "constraint": "type",
                    "current": content_type,
                    "suggested": "both",
                    "reason": f"Consider {other_type} as well for more options",
                    "priority": 5
                })
                priority_order.append(5)

            if 'genre' in requirements:
                suggestions.append({
                    "constraint": "genre",
                    "current": requirements['genre'],
                    "suggested": "remove",
                    "reason": "Remove genre restriction to find more mood-matching content",
                    "priority": 6
                })
                priority_order.append(6)

            # Sort suggestions by priority
            suggestions.sort(key=lambda x: x['priority'])

            return {
                "success": True,
                "current_results": results_count,
                "suggestions": suggestions,
                "count": len(suggestions)
            }

        except Exception as e:
            logger.error(f"Constraint relaxer error: {e}")
            return {"success": False, "error": str(e), "suggestions": [], "count": 0}

    def as_langchain_tool(self) -> StructuredTool:
        def run_tool(requirements: Dict[str, Any] = {}, results_count: int = 0) -> str:
            try:
                result = self.suggest_relaxations(requirements, results_count)

                if result['success']:
                    output = f"Current search returned {result['current_results']} results\n\n"

                    if result['count'] > 0:
                        output += f"Suggested relaxations (in priority order):\n\n"
                        for i, suggestion in enumerate(result['suggestions'], 1):
                            output += f"{i}. {suggestion['constraint'].upper()}\n"
                            output += f"   Current: {suggestion['current']}\n"
                            output += f"   Suggested: {suggestion['suggested']}\n"
                            output += f"   Reason: {suggestion['reason']}\n\n"
                    else:
                        output += "No specific relaxation suggestions available."

                    return output
                else:
                    return f"Error: {result.get('error', 'Unknown error')}"

            except Exception as e:
                return f"Error: {str(e)}"

        return StructuredTool.from_function(
            func=run_tool,
            name="constraint_relaxer",
            description="Suggests how to relax constraints when searches return too few results. Use this when other tools return insufficient recommendations.",
            args_schema=ConstraintRelaxerInput
        )


def create_constraint_relaxer_tool(vector_store) -> StructuredTool:
    relaxer = ConstraintRelaxer(vector_store)
    return relaxer.as_langchain_tool()
