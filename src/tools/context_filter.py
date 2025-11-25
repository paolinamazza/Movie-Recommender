"""
Tool: Context Filter

Filters content based on viewing context (solo, family, date night, etc.).
"""

import json
import logging
from typing import List, Dict, Any
from langchain.tools import StructuredTool
from pydantic.v1 import BaseModel, Field
from src.utils.request_context import get_items, set_items

from src.utils.helpers import parse_metadata_field

logger = logging.getLogger(__name__)


class ContextFilterInput(BaseModel):
    """Input schema for context filter tool."""
    context: str = Field(description="Viewing context: solo_watch, family_watch, date_night, background_watch, or binge_worthy")
    items: List[Dict[str, Any]] = Field(default_factory=list, description="Optional list of content items to filter. If empty, searches database directly.")
    strict: bool = Field(default=False, description="If True, only return items with suitability >= 0.7; if False, >= 0.5")


class ContextFilter:
    """Filters content based on viewing context suitability."""

    def __init__(self, vector_store):
        """
        Initialize the context filter.

        Args:
            vector_store: VectorStoreManager instance
        """
        self.vector_store = vector_store
        self.valid_contexts = [
            "solo_watch", "family_watch", "date_night",
            "background_watch", "binge_worthy"
        ]

    def filter(self, context: str, items: List[Dict[str, Any]] = None, strict: bool = False) -> Dict[str, Any]:
        """
        Filter items by viewing context suitability.

        Args:
            context: Target viewing context
            items: Optional list of content items (can be IDs or full items). If None/empty, searches database.
            strict: If True, use 0.7 threshold; if False, use 0.5

        Returns:
            Dictionary with filtered items and metadata
        """
        # If no items provided, try to get from context
        if not items:
            items = get_items()
            if items:
                logger.info(f"Using {len(items)} items from context")

        # If still no items, search database for context-suitable content
        if not items:
            logger.info(f"No items provided, searching database for context={context}")
            try:
                # Search database with context-related query
                context_queries = {
                    "family_watch": "family-friendly kid-friendly appropriate for all ages wholesome",
                    "date_night": "romantic comedy thriller suspense engaging entertaining",
                    "solo_watch": "thought-provoking deep complex mature engaging",
                    "background_watch": "light relaxing casual easy-watching comfortable",
                    "binge_worthy": "addictive compelling series multiple seasons engaging"
                }

                query = context_queries.get(context, "popular highly rated")
                results = self.vector_store.search(query=query, n_results=50)

                # Convert search results to items
                if results['metadatas'] and len(results['metadatas'][0]) > 0:
                    items = [self._format_item(m) for m in results['metadatas'][0]]
                else:
                    return {
                        "success": False,
                        "error": f"No content found for context={context}",
                        "items": [],
                        "count": 0
                    }
            except Exception as e:
                logger.error(f"Database search error: {e}")
                return {
                    "success": False,
                    "error": f"Database search failed: {str(e)}",
                    "items": [],
                    "count": 0
                }

        logger.info(f"Filtering {len(items)} items for context={context}, strict={strict}")

        if context not in self.valid_contexts:
            return {
                "success": False,
                "error": f"Invalid context. Must be one of: {', '.join(self.valid_contexts)}",
                "items": [],
                "count": 0
            }

        threshold = 0.7 if strict else 0.5
        filtered_items = []

        try:
            for item in items:
                # If item is just an ID, fetch full metadata
                if isinstance(item, (str, int)):
                    results = self.vector_store.get_by_ids([str(item)])
                    if results['metadatas']:
                        metadata = results['metadatas'][0]
                        item = self._format_item(metadata)
                    else:
                        continue

                # Get context suitability score
                suitability = item.get('context_suitability', {})
                if isinstance(suitability, str):
                    suitability = json.loads(suitability)

                score = suitability.get(context, 0)

                if score >= threshold:
                    item['context_suitability_score'] = score
                    filtered_items.append(item)

            # Sort by suitability score
            filtered_items.sort(key=lambda x: x.get('context_suitability_score', 0), reverse=True)

            logger.info(f"Filtered to {len(filtered_items)} items (threshold={threshold})")
            
            # Update context
            set_items(filtered_items)

            return {
                "success": True,
                "context": context,
                "threshold": threshold,
                "items": filtered_items,
                "count": len(filtered_items),
                "original_count": len(items)
            }

        except Exception as e:
            logger.error(f"Context filter error: {e}")
            return {
                "success": False,
                "error": str(e),
                "items": [],
                "count": 0
            }

    def _format_item(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Format metadata into item dictionary."""
        return {
            "id": metadata.get("content_id"),
            "type": metadata.get("type"),
            "title": metadata.get("title"),
            "year": metadata.get("year"),
            "tmdb_rating": metadata.get("tmdb_rating"),
            "genres": parse_metadata_field(metadata, "genres"),
            "moods": parse_metadata_field(metadata, "moods"),
            "runtime_minutes": metadata.get("runtime_minutes"),
            "total_hours": metadata.get("total_hours"),
            "content_rating": metadata.get("content_rating"),
            "context_suitability": parse_metadata_field(metadata, "context_suitability"),
            "cast": parse_metadata_field(metadata, "cast"),
            "crew": parse_metadata_field(metadata, "crew"),
        }

    def as_langchain_tool(self) -> StructuredTool:
        """
        Create a LangChain StructuredTool wrapper.

        Returns:
            LangChain StructuredTool instance
        """
        def run_tool(context: str, items: List[Dict[str, Any]] = None, strict: bool = False) -> str:
            """Execute the tool with structured input."""
            try:
                result = self.filter(context, items, strict)

                # Format output
                if result['success']:
                    output = f"Filtered {result['original_count']} items for context '{context}' "
                    output += f"(threshold={result['threshold']})\n"
                    output += f"Result: {result['count']} suitable items\n\n"

                    if result['items']:
                        output += "Top suitable items:\n"
                        for i, item in enumerate(result['items'][:5], 1):
                            output += f"{i}. {item['title']} ({item['year']}) - {item['type']}\n"
                            output += f"   Suitability: {item.get('context_suitability_score', 0):.2f}, "
                            output += f"Rating: {item['tmdb_rating']}/10\n"
                    else:
                        output += "No items meet the suitability threshold."

                    # ALWAYS output JSON for machine parsing
                    output += f"\n__ITEMS_JSON__: {json.dumps(result['items'])}"

                    return output
                else:
                    return f"Error: {result.get('error', 'Unknown error')}"

            except Exception as e:
                return f"Error executing tool: {str(e)}"

        return StructuredTool.from_function(
            func=run_tool,
            name="context_filter",
            description="Filters content items based on viewing context suitability. Use this when users specify a viewing situation (solo, family, date night, background, binge).",
            args_schema=ContextFilterInput
        )


def create_context_filter_tool(vector_store) -> StructuredTool:
    """
    Factory function to create the context filter tool.

    Args:
        vector_store: VectorStoreManager instance

    Returns:
        LangChain StructuredTool
    """
    filter_tool = ContextFilter(vector_store)
    return filter_tool.as_langchain_tool()
