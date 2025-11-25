"""
Tool: Mood Content Finder

Finds movies/series matching a specific mood with minimum rating threshold.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from langchain.tools import StructuredTool
from pydantic.v1 import BaseModel, Field

from src.utils.helpers import parse_metadata_field, format_content_item
from src.utils.request_context import set_items, get_excluded_items

logger = logging.getLogger(__name__)


class MoodContentFinderInput(BaseModel):
    """Input schema for mood content finder tool."""
    mood: str = Field(description="Target mood: happy, sad, excited, relaxed, scared, curious, inspired, stressed, nostalgic, or bored")
    content_type: Optional[str] = Field(default=None, description="Content type filter: 'movie', 'series', or None for both")
    min_rating: float = Field(default=6.0, description="Minimum TMDB rating (0-10)")


class MoodContentFinder:
    """Finds content matching specific moods."""

    def __init__(self, vector_store):
        """
        Initialize the mood content finder.

        Args:
            vector_store: VectorStoreManager instance
        """
        self.vector_store = vector_store

    def find(self, mood: str, content_type: Optional[str] = None, min_rating: float = 6.0) -> Dict[str, Any]:
        """
        Find content matching the specified mood.

        Args:
            mood: Target emotional mood
            content_type: Optional type filter ("movie" or "series")
            min_rating: Minimum rating threshold

        Returns:
            Dictionary with matched items and metadata
        """
        logger.info(f"Finding {content_type or 'content'} with mood={mood}, min_rating={min_rating}")

        mood = mood.lower()
        valid_moods = ["happy", "sad", "excited", "relaxed", "scared", "curious",
                       "inspired", "stressed", "nostalgic", "bored"]

        if mood not in valid_moods:
            return {
                "success": False,
                "error": f"Invalid mood. Must be one of: {', '.join(valid_moods)}",
                "items": [],
                "count": 0
            }

        # Build metadata filter
        where_filter = {}
        if content_type in ['movie', 'series']:
            where_filter['type'] = content_type

        # Search with mood-related query
        mood_queries = {
            "happy": "uplifting cheerful feel-good comedy joyful",
            "sad": "emotional tear-jerking melancholic dramatic tragic",
            "excited": "thrilling action-packed adrenaline intense fast-paced",
            "relaxed": "calm soothing peaceful meditative gentle",
            "scared": "scary frightening suspenseful horror terrifying",
            "curious": "mysterious intriguing thought-provoking investigative",
            "inspired": "motivational uplifting biographical achievement inspirational",
            "stressed": "intense gripping nerve-wracking high-stakes",
            "nostalgic": "sentimental reminiscent vintage period coming-of-age",
            "bored": "entertaining engaging escapist adventure fantasy"
        }

        query = mood_queries.get(mood, mood)

        try:
            results = self.vector_store.search(
                query=query,
                n_results=50,  # Get more results to filter
                where=where_filter if where_filter else None
            )

            # Filter and format results
            items = []
            fallback_used = False
            if results['ids'] and len(results['ids']) > 0:
                for metadata in results['metadatas'][0]:
                    item = format_content_item(metadata, include_all=True)

                    # Check mood match
                    item_moods = [m.lower() for m in item.get('moods', [])]
                    if mood not in item_moods:
                        continue

                    # Check rating
                    if item.get('tmdb_rating', 0) < min_rating:
                        continue

                    # Add confidence score
                    item['mood_match_confidence'] = 1.0 if mood in item_moods else 0.0

                    items.append(item)

            # Fallback: if no items match the exact mood metadata, return top results anyway
            if not items and results['ids'] and len(results['ids']) > 0:
                fallback_used = True
                for metadata in results['metadatas'][0]:
                    item = format_content_item(metadata, include_all=True)
                    item['mood_match_confidence'] = 0.0
                    items.append(item)

            # Sort by rating
            items.sort(key=lambda x: x.get('tmdb_rating', 0), reverse=True)

            # Filter out globally excluded items
            excluded_ids = set(get_excluded_items())
            if excluded_ids:
                original_count = len(items)
                items = [item for item in items if str(item.get('id')) not in excluded_ids]
                if len(items) < original_count:
                    logger.info(f"Filtered {original_count - len(items)} globally excluded items")

            # Limit to top 10
            items = items[:10]

            logger.info(f"Found {len(items)} items matching mood={mood}")
            
            # Update context for subsequent tools
            set_items(items)

            return {
                "success": True,
                "mood": mood,
                "content_type": content_type,
                "min_rating": min_rating,
                "items": items,
                "count": len(items),
                "fallback_used": fallback_used
            }

        except Exception as e:
            logger.error(f"Mood content finder error: {e}")
            return {
                "success": False,
                "error": str(e),
                "items": [],
                "count": 0
            }

    def as_langchain_tool(self) -> StructuredTool:
        """
        Create a LangChain StructuredTool wrapper.

        Returns:
            LangChain StructuredTool instance
        """
        def run_tool(mood: str, content_type: Optional[str] = None, min_rating: float = 6.0) -> str:
            """Execute the tool with structured input."""
            try:
                result = self.find(mood, content_type, min_rating)

                # Format output
                if result['success']:
                    output = f"Found {result['count']} items matching mood '{mood}'"
                    if content_type:
                        output += f" (type: {content_type})"
                    output += f" with rating >= {min_rating}\n\n"
                    if result.get('fallback_used'):
                        output += "(Fallback applied: insufficient explicit mood metadata, showing best overall matches.)\n\n"

                    if result['items']:
                        output += "Top matches:\n"
                        for i, item in enumerate(result['items'][:5], 1):
                            output += f"{i}. {item['title']} ({item['year']}) - {item['type']}\n"
                            output += f"   Rating: {item['tmdb_rating']}/10, Moods: {', '.join(item['moods'][:3])}\n"
                    else:
                        output += "No items found matching criteria."

                    # ALWAYS output JSON for machine parsing
                    output += f"\n__ITEMS_JSON__: {json.dumps(result['items'])}"

                    return output
                else:
                    return f"Error: {result.get('error', 'Unknown error')}"

            except Exception as e:
                return f"Error executing tool: {str(e)}"

        return StructuredTool.from_function(
            func=run_tool,
            name="mood_content_finder",
            description="Finds movies or series that match a specific emotional mood. Use this tool FIRST for any mood or genre-related queries (happy, sad, scary, funny, etc.).",
            args_schema=MoodContentFinderInput
        )


def create_mood_content_finder_tool(vector_store) -> StructuredTool:
    """
    Factory function to create the mood content finder tool.

    Args:
        vector_store: VectorStoreManager instance

    Returns:
        LangChain Tool
    """
    finder = MoodContentFinder(vector_store)
    return finder.as_langchain_tool()
