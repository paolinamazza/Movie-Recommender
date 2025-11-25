"""
Tool: Viewing History Checker

Checks items against user viewing history to avoid repetition.
"""

import json
import logging
from typing import List, Dict, Any
from langchain.tools import StructuredTool
from pydantic.v1 import BaseModel, Field
from src.utils.request_context import get_items, set_items, add_excluded_items

logger = logging.getLogger(__name__)


class ViewingHistoryInput(BaseModel):
    """Input schema for viewing history checker tool."""
    user_history: List[str] = Field(default=[], description="List of previously watched titles or IDs")
    items: List[Dict[str, Any]] = Field(default=[], description="List of content items to filter. REQUIRED - must get items from a previous tool call (e.g., mood_content_finder, runtime_matcher, etc.) before using this tool.")
    avoid_similar: bool = Field(default=True, description="If True, also avoid similar genres/moods")


class ViewingHistoryChecker:
    """Checks content against viewing history."""

    def __init__(self, vector_store):
        self.vector_store = vector_store

    def check(self, user_history: List[str], items: List[Dict[str, Any]] = None, avoid_similar: bool = True) -> Dict[str, Any]:
        """
        Filter items based on viewing history.

        Args:
            user_history: List of previously watched titles or IDs
            items: Optional list of content items. If None/empty, will search vector store directly.
            avoid_similar: If True, also avoid similar genres/moods

        Returns:
            Filtered items avoiding history
        """
        # If no items provided, try to get from context
        if not items or len(items) == 0:
            items = get_items()
            if items:
                logger.info(f"Using {len(items)} items from context")

        # If still no items, search vector store directly
        if not items or len(items) == 0:
            if not user_history:
                return {
                    "success": False,
                    "error": "No user history provided and no items to filter",
                    "items": [],
                    "count": 0
                }

            # Search for fresh content by using general query
            logger.info(f"No items provided, searching vector store for content avoiding: {user_history}")
            try:
                search_results = self.vector_store.search(query="movie", n_results=50)
                if search_results and 'metadatas' in search_results and len(search_results['metadatas']) > 0:
                    items = []
                    for metadata in search_results['metadatas'][0]:
                        items.append(metadata)
                else:
                    return {
                        "success": False,
                        "error": "Could not find any content in vector store",
                        "items": [],
                        "count": 0
                    }
            except Exception as e:
                logger.error(f"Error searching vector store: {e}")
                return {
                    "success": False,
                    "error": f"Error searching: {str(e)}",
                    "items": [],
                    "count": 0
                }

        logger.info(f"Checking {len(items)} items against history of {len(user_history)} items")

        history_titles = [h.lower() for h in user_history if isinstance(h, str)]
        filtered_items = []

        try:
            # Get genres/moods from history for similarity check
            history_genres = set()
            history_moods = set()

            if avoid_similar and user_history:
                # Search for history items to get their metadata
                for hist_item in user_history[:5]:  # Check last 5 items
                    try:
                        results = self.vector_store.search(query=str(hist_item), n_results=1)
                        if results['metadatas'] and len(results['metadatas'][0]) > 0:
                            metadata = results['metadatas'][0][0]
                            genres = json.loads(metadata.get('genres', '[]'))
                            moods = json.loads(metadata.get('moods', '[]'))
                            history_genres.update([g.lower() for g in genres])
                            history_moods.update([m.lower() for m in moods])
                    except:
                        pass

            for item in items:
                title = item.get('title', '').lower()
                item_id = str(item.get('id', ''))

                # Skip if exact match in history
                if title in history_titles or item_id in user_history:
                    continue

                # Check similarity if enabled
                if avoid_similar:
                    item_genres = [g.lower() for g in item.get('genres', [])]
                    item_moods = [m.lower() for m in item.get('moods', [])]

                    genre_overlap = len(set(item_genres) & history_genres)
                    mood_overlap = len(set(item_moods) & history_moods)

                    # Calculate novelty score
                    total_possible = max(len(item_genres) + len(item_moods), 1)
                    overlap = genre_overlap + mood_overlap
                    novelty_score = 1.0 - (overlap / total_possible)

                    item['novelty_score'] = novelty_score

                    # Skip if too similar (>95% overlap, essentially duplicates or near-duplicates)
                    # Changed from 0.2 to 0.05 to be less aggressive
                    if novelty_score < 0.05:
                        continue

                filtered_items.append(item)

            # Sort by novelty if avoid_similar is True
            if avoid_similar:
                filtered_items.sort(key=lambda x: x.get('novelty_score', 0.5), reverse=True)

            # Identify removed items and add to exclusion list
            removed_ids = []
            filtered_ids = {str(item.get('id')) for item in filtered_items}
            for item in items:
                item_id = str(item.get('id'))
                if item_id not in filtered_ids:
                    removed_ids.append(item_id)
            
            if removed_ids:
                add_excluded_items(removed_ids)
                logger.info(f"Added {len(removed_ids)} items to global exclusion list")

            logger.info(f"Filtered to {len(filtered_items)} novel items")
            
            # Update context
            set_items(filtered_items)

            return {
                "success": True,
                "items": filtered_items,
                "count": len(filtered_items),
                "removed_count": len(items) - len(filtered_items)
            }

        except Exception as e:
            logger.error(f"History checker error: {e}")
            return {"success": False, "error": str(e), "items": [], "count": 0}

    def as_langchain_tool(self) -> StructuredTool:
        def run_tool(user_history: List[str] = [], items: List[Dict[str, Any]] = [], avoid_similar: bool = True) -> str:
            try:
                result = self.check(user_history, items, avoid_similar)

                if result['success']:
                    output = f"Checked {len(items)} items against history\n"
                    output += f"Removed {result['removed_count']} similar/watched items\n"
                    output += f"Remaining: {result['count']} novel recommendations\n\n"

                    if result['items']:
                        output += "Top novel items:\n"
                        for i, item in enumerate(result['items'][:5], 1):
                            output += f"{i}. {item['title']} ({item['year']})\n"
                            output += f"   Novelty: {item.get('novelty_score', 0):.2f}\n"
                    else:
                        output += "No items meet the criteria."

                    # ALWAYS output JSON for machine parsing
                    output += f"\n__ITEMS_JSON__: {json.dumps(result['items'])}"

                    return output
                else:
                    return f"Error: {result.get('error', 'Unknown error')}"

            except Exception as e:
                return f"Error: {str(e)}"

        return StructuredTool.from_function(
            func=run_tool,
            name="viewing_history_checker",
            description="Filters a list of items to avoid previously watched content. IMPORTANT: This tool requires 'items' from a previous tool call (like mood_content_finder or runtime_matcher). Do NOT call this tool first - always get items from another tool before filtering by viewing history. Use when users mention what they've already watched.",
            args_schema=ViewingHistoryInput
        )


def create_viewing_history_checker_tool(vector_store) -> StructuredTool:
    checker = ViewingHistoryChecker(vector_store)
    return checker.as_langchain_tool()
