"""
Tool: General Search Tool

Performs a semantic search on the movie database using the vector store.
Useful for queries that don't fit into specific mood or similarity categories.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from langchain.tools import StructuredTool
from pydantic.v1 import BaseModel, Field

from src.utils.helpers import format_content_item
from src.utils.request_context import set_items, get_excluded_items

logger = logging.getLogger(__name__)


class GeneralSearchInput(BaseModel):
    """Input schema for general search tool."""
    query: str = Field(description="The search query string (e.g., 'space movies', '90s action', 'movies about cooking')")
    n: int = Field(default=5, description="Number of results to return")


class GeneralSearchTool:
    """Performs semantic search on the content database."""

    def __init__(self, vector_store):
        self.vector_store = vector_store

    def search(self, query: str, n: int = 5) -> Dict[str, Any]:
        """
        Execute semantic search.

        Args:
            query: Search query
            n: Number of results

        Returns:
            Search results
        """
        logger.info(f"General search for: '{query}'")

        try:
            results = self.vector_store.search(query=query, n_results=n)

            items = []
            if results['ids'] and len(results['ids']) > 0:
                for i, metadata in enumerate(results['metadatas'][0]):
                    item = format_content_item(metadata, include_all=True)
                    
                    # Add distance/score if available
                    if results.get('distances'):
                        item['distance'] = results['distances'][0][i]
                    
                    items.append(item)

            # Filter out globally excluded items
            excluded_ids = set(get_excluded_items())
            if excluded_ids:
                original_count = len(items)
                items = [item for item in items if str(item.get('id')) not in excluded_ids]
                if len(items) < original_count:
                    logger.info(f"Filtered {original_count - len(items)} globally excluded items")

            # Update context
            set_items(items)
            
            return {
                "success": True,
                "query": query,
                "items": items,
                "count": len(items)
            }

        except Exception as e:
            logger.error(f"General search error: {e}")
            return {
                "success": False,
                "error": str(e),
                "items": [],
                "count": 0
            }

    def as_langchain_tool(self) -> StructuredTool:
        def run_tool(query: str, n: int = 5) -> str:
            try:
                result = self.search(query, n)

                if result['success']:
                    output = f"Found {result['count']} items matching '{query}':\n\n"
                    
                    if result['items']:
                        output += "Top search results:\n"
                        for i, item in enumerate(result['items'][:5], 1):
                            output += f"{i}. {item['title']} ({item['year']}) - {item['type']}\n"
                            output += f"   Relevance: {item.get('relevance_score', 0):.2f}\n"
                    else:
                        output += "No items found matching the search query."

                    # ALWAYS output JSON for machine parsing
                    output += f"\n__ITEMS_JSON__: {json.dumps(result['items'])}"

                    return output
                else:
                    return f"Error: {result.get('error', 'Unknown error')}"

            except Exception as e:
                return f"Error: {str(e)}"

        return StructuredTool.from_function(
            func=run_tool,
            name="general_search_tool",
            description="Performs a general semantic search for movies/series. Use this when the user's request doesn't fit specific moods or isn't about a specific similar title (e.g., 'movies about space', '90s action', 'plot with twists').",
            args_schema=GeneralSearchInput
        )


def create_general_search_tool(vector_store) -> StructuredTool:
    tool = GeneralSearchTool(vector_store)
    return tool.as_langchain_tool()
