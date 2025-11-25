"""
ConversationMemory: Tracks conversation history and user preferences.
"""

import logging
from typing import List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    timestamp: datetime
    query: str
    recommendations: List[str]
    strategy_used: str
    user_feedback: str = ""


class ConversationMemory:
    """Maintains conversation history and learned preferences."""

    def __init__(self):
        self.turns: List[ConversationTurn] = []
        self.user_preferences: Dict[str, Any] = {}

    def add_turn(self, query: str, recommendations: List[str], strategy: str):
        """Add a conversation turn."""
        turn = ConversationTurn(
            timestamp=datetime.now(),
            query=query,
            recommendations=recommendations,
            strategy_used=strategy
        )
        self.turns.append(turn)
        logger.info(f"Added turn {len(self.turns)}: {query[:50]}...")

    def get_recent_context(self, n: int = 3) -> List[Dict]:
        """Get recent conversation turns for context."""
        recent = self.turns[-n:] if len(self.turns) > 0 else []
        return [
            {
                "query": turn.query,
                "recommendations_count": len(turn.recommendations),
                "strategy": turn.strategy_used
            }
            for turn in recent
        ]

    def get_watched_titles(self) -> List[str]:
        """Get all titles from previous recommendations (implies watched/seen)."""
        watched = []
        for turn in self.turns:
            watched.extend(turn.recommendations)
        return list(set(watched))

    def clear(self):
        """Clear conversation history."""
        self.turns = []
        self.user_preferences = {}
        logger.info("Cleared conversation memory")
