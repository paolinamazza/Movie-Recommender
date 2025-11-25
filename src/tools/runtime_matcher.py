"""
Tool: Runtime Matcher

Matches content to available viewing time and session type.
"""

import json
import logging
from typing import List, Dict, Any
from langchain.tools import StructuredTool
from pydantic.v1 import BaseModel, Field
from src.utils.request_context import get_items, set_items

logger = logging.getLogger(__name__)


class RuntimeMatcherInput(BaseModel):
    """Input schema for runtime matcher tool."""
    available_hours: float = Field(description="Available viewing time in hours")
    items: List[Dict[str, Any]] = Field(default_factory=list, description="Optional list of content items to match. If empty, searches database directly.")
    session_type: str = Field(default="single", description="Session type: 'single' for one sitting, 'multiple' for series over multiple sessions")


class RuntimeMatcher:
    """Matches content based on runtime and available time."""

    def __init__(self, vector_store):
        """
        Initialize the runtime matcher.

        Args:
            vector_store: VectorStoreManager instance
        """
        self.vector_store = vector_store

    def match(self, available_hours: float, items: List[Dict[str, Any]] = None, session_type: str = "single") -> Dict[str, Any]:
        """
        Match items to available viewing time.

        Args:
            available_hours: Available time in hours
            items: Optional list of content items. If None/empty, searches database.
            session_type: "single" for movies/single episodes, "multiple" for series

        Returns:
            Dictionary with matched items and fit quality
        """
        # If no items provided, try to get from context
        if not items:
            items = get_items()
            if items:
                logger.info(f"Using {len(items)} items from context")

        # If still no items, search database
        if not items:
            logger.info(f"No items provided, searching database for {available_hours}h runtime")
            try:
                from src.utils.helpers import format_content_item

                # Search for popular content
                results = self.vector_store.search(query="popular highly rated", n_results=100)

                # Convert search results to items
                if results['metadatas'] and len(results['metadatas'][0]) > 0:
                    items = [format_content_item(m, include_all=True) for m in results['metadatas'][0]]
                else:
                    return {
                        "success": False,
                        "error": "No content found in database",
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

        logger.info(f"Matching {len(items)} items to {available_hours}h ({session_type} session)")

        if available_hours <= 0:
            return {
                "success": False,
                "error": "Available hours must be positive",
                "items": [],
                "count": 0
            }

        matched_items = []

        try:
            for item in items:
                total_hours = item.get('total_hours', 0)
                content_type = item.get('type', 'movie')
                runtime_minutes = item.get('runtime_minutes', 0)

                fit_quality = 0.0
                match_reason = ""

                if session_type == "single":
                    # For single session, prefer content that fits well
                    if content_type == "movie":
                        if total_hours <= available_hours:
                            # Perfect fit or good fit
                            if total_hours >= available_hours * 0.7:
                                fit_quality = 1.0
                                match_reason = "Perfect fit for available time"
                            else:
                                fit_quality = 0.8
                                match_reason = "Good fit with time to spare"
                        else:
                            # Too long
                            fit_quality = 0.3
                            match_reason = "Slightly too long"
                    else:
                        # Series: check if single episode fits
                        episode_hours = runtime_minutes / 60.0 if runtime_minutes else 0.75
                        if episode_hours <= available_hours:
                            fit_quality = 0.9
                            match_reason = "Single episode fits well"
                        else:
                            fit_quality = 0.2
                            match_reason = "Episodes too long"

                elif session_type == "multiple":
                    # For multiple sessions, prefer series
                    if content_type == "series":
                        seasons = item.get('seasons', 1)
                        episodes = item.get('episodes', 1)
                        episode_hours = runtime_minutes / 60.0 if runtime_minutes else 0.75

                        # Check if episodes fit in typical sessions
                        if episode_hours <= available_hours:
                            # Good bingeable series
                            if seasons >= 2 and episodes >= 10:
                                fit_quality = 1.0
                                match_reason = "Great for multiple sessions"
                            else:
                                fit_quality = 0.7
                                match_reason = "Suitable for binge watching"
                        else:
                            fit_quality = 0.4
                            match_reason = "Episodes longer than typical session"
                    else:
                        # Movies are less ideal for multiple sessions
                        fit_quality = 0.5
                        match_reason = "Movie (better for single session)"

                if fit_quality > 0:
                    item['runtime_fit_quality'] = fit_quality
                    item['runtime_match_reason'] = match_reason
                    item['fits_in_time'] = total_hours <= available_hours
                    matched_items.append(item)

            # Sort by fit quality
            matched_items.sort(key=lambda x: x.get('runtime_fit_quality', 0), reverse=True)

            logger.info(f"Matched {len(matched_items)} items to available time")
            
            # Update context
            set_items(matched_items)

            return {
                "success": True,
                "available_hours": available_hours,
                "session_type": session_type,
                "items": matched_items,
                "count": len(matched_items)
            }

        except Exception as e:
            logger.error(f"Runtime matcher error: {e}")
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
        def run_tool(available_hours: float, items: List[Dict[str, Any]] = None, session_type: str = "single") -> str:
            """Execute the tool with structured input."""
            try:
                result = self.match(available_hours, items, session_type)

                # Format output
                if result['success']:
                    output = f"Matched {result['count']} items to {available_hours} hours "
                    output += f"(session: {session_type})\n\n"

                    if result['items']:
                        output += "Best matches:\n"
                        for i, item in enumerate(result['items'][:5], 1):
                            output += f"{i}. {item['title']} ({item['year']}) - {item['type']}\n"
                            output += f"   Runtime: {item.get('total_hours', 0):.1f}h, Fit Quality: {item.get('runtime_fit_quality', 0):.2f}\n"
                            output += f"   {item.get('runtime_match_reason', '')}\n"
                    else:
                        output += "No items fit the runtime constraints."

                    # ALWAYS output JSON for machine parsing
                    output += f"\n__ITEMS_JSON__: {json.dumps(result['items'])}"

                    return output
                else:
                    return f"Error: {result.get('error', 'Unknown error')}"

            except Exception as e:
                return f"Error executing tool: {str(e)}"

        return StructuredTool.from_function(
            func=run_tool,
            name="runtime_matcher",
            description="Matches content items to available viewing time. Use this when users specify time constraints (e.g., 'under 2 hours', 'quick watch').",
            args_schema=RuntimeMatcherInput
        )


def create_runtime_matcher_tool(vector_store) -> StructuredTool:
    """
    Factory function to create the runtime matcher tool.

    Args:
        vector_store: VectorStoreManager instance

    Returns:
        LangChain Tool
    """
    matcher = RuntimeMatcher(vector_store)
    return matcher.as_langchain_tool()
