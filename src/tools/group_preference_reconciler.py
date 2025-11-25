"""
Tool: Group Preference Reconciler

Reconciles preferences from multiple users to find compromise recommendations.
"""

import json
import logging
from typing import List, Dict, Any
from langchain.tools import StructuredTool
from pydantic.v1 import BaseModel, Field

logger = logging.getLogger(__name__)


class GroupPreferenceInput(BaseModel):
    """Input schema for group preference reconciler."""
    preferences_list: List[Dict[str, Any]] = Field(
        default=[],
        description="List of preference dictionaries, one per user. Each dict can contain: mood, genre, context, min_rating, max_hours, type"
    )


class GroupPreferenceReconciler:
    """Reconciles multiple user preferences for group viewing."""

    def __init__(self, vector_store):
        self.vector_store = vector_store

    def reconcile(self, preferences_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Find compromise preferences from multiple users.

        Args:
            preferences_list: List of preference dictionaries, one per user

        Returns:
            Reconciled preferences and search strategy
        """
        logger.info(f"Reconciling preferences from {len(preferences_list)} users")

        if len(preferences_list) == 0:
            return {
                "success": False,
                "error": "No preferences provided",
                "reconciled": {},
                "strategy": ""
            }

        try:
            # Collect all moods
            all_moods = []
            for prefs in preferences_list:
                if 'mood' in prefs:
                    all_moods.append(prefs['mood'].lower())

            # Collect all genres
            all_genres = []
            for prefs in preferences_list:
                if 'genre' in prefs:
                    all_genres.append(prefs['genre'].lower())

            # Collect contexts
            all_contexts = []
            for prefs in preferences_list:
                if 'context' in prefs:
                    all_contexts.append(prefs['context'])

            # Collect rating preferences
            min_ratings = []
            for prefs in preferences_list:
                if 'min_rating' in prefs:
                    min_ratings.append(float(prefs['min_rating']))

            # Collect time constraints
            max_hours_list = []
            for prefs in preferences_list:
                if 'max_hours' in prefs:
                    max_hours_list.append(float(prefs['max_hours']))

            # Reconciliation strategy
            reconciled = {}
            strategy_notes = []

            # For moods: Find common ground or complementary moods
            if all_moods:
                mood_counts = {}
                for mood in all_moods:
                    mood_counts[mood] = mood_counts.get(mood, 0) + 1

                # If there's a consensus mood (>50%), use it
                max_count = max(mood_counts.values())
                if max_count > len(preferences_list) / 2:
                    consensus_moods = [m for m, c in mood_counts.items() if c == max_count]
                    reconciled['moods'] = consensus_moods
                    strategy_notes.append(f"Consensus mood: {', '.join(consensus_moods)}")
                else:
                    # Use multiple moods that balance preferences
                    reconciled['moods'] = list(mood_counts.keys())
                    strategy_notes.append(f"Balanced moods: {', '.join(mood_counts.keys())}")

            # For genres: Similar to moods
            if all_genres:
                genre_counts = {}
                for genre in all_genres:
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1

                if len(genre_counts) == 1:
                    reconciled['genres'] = list(genre_counts.keys())
                    strategy_notes.append(f"Agreed genre: {list(genre_counts.keys())[0]}")
                else:
                    # Include all requested genres
                    reconciled['genres'] = list(genre_counts.keys())
                    strategy_notes.append(f"Multiple genres: {', '.join(genre_counts.keys())}")

            # For context: Default to most common, or family_watch if no consensus
            if all_contexts:
                context_counts = {}
                for ctx in all_contexts:
                    context_counts[ctx] = context_counts.get(ctx, 0) + 1

                most_common_context = max(context_counts, key=context_counts.get)
                reconciled['context'] = most_common_context
                strategy_notes.append(f"Context: {most_common_context}")
            else:
                reconciled['context'] = 'family_watch'
                strategy_notes.append("Default context: family_watch for group viewing")

            # For rating: Use the maximum (satisfy the pickiest viewer)
            if min_ratings:
                reconciled['min_rating'] = max(min_ratings)
                strategy_notes.append(f"Using highest rating requirement: {max(min_ratings)}")

            # For time: Use the minimum (fit everyone's schedule)
            if max_hours_list:
                reconciled['max_hours'] = min(max_hours_list)
                strategy_notes.append(f"Using shortest time constraint: {min(max_hours_list)}h")

            # Type preference
            types = []
            for prefs in preferences_list:
                if 'type' in prefs:
                    types.append(prefs['type'])

            if types and len(set(types)) == 1:
                reconciled['type'] = types[0]
                strategy_notes.append(f"Agreed type: {types[0]}")

            strategy = "Group recommendation strategy:\n" + "\n".join(f"- {note}" for note in strategy_notes)

            return {
                "success": True,
                "user_count": len(preferences_list),
                "reconciled": reconciled,
                "strategy": strategy,
                "original_preferences": preferences_list
            }

        except Exception as e:
            logger.error(f"Group preference reconciler error: {e}")
            return {"success": False, "error": str(e), "reconciled": {}, "strategy": ""}

    def as_langchain_tool(self) -> StructuredTool:
        def run_tool(preferences_list: List[Dict[str, Any]] = []) -> str:
            try:
                result = self.reconcile(preferences_list)

                if result['success']:
                    output = f"Reconciled preferences from {result['user_count']} users\n\n"
                    output += f"{result['strategy']}\n\n"
                    output += f"Reconciled Search Parameters:\n"
                    output += json.dumps(result['reconciled'], indent=2)

                    return output
                else:
                    return f"Error: {result.get('error', 'Unknown error')}"

            except Exception as e:
                return f"Error: {str(e)}"

        return StructuredTool.from_function(
            func=run_tool,
            name="group_preference_reconciler",
            description="Reconciles preferences from multiple users to find compromise recommendations for group viewing. Use this when multiple people with different preferences need to agree on what to watch.",
            args_schema=GroupPreferenceInput
        )


def create_group_preference_reconciler_tool(vector_store) -> StructuredTool:
    reconciler = GroupPreferenceReconciler(vector_store)
    return reconciler.as_langchain_tool()
