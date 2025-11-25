"""
Tool: Similar Content Finder

Finds content similar to a given item.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from langchain.tools import StructuredTool
from pydantic.v1 import BaseModel, Field
from src.utils.helpers import format_content_item
from src.utils.request_context import set_items

logger = logging.getLogger(__name__)


class SimilarContentInput(BaseModel):
    """Input schema for similar content finder."""
    item_id: str = Field(description="ID or title of the reference item")
    n: int = Field(default=5, description="Number of similar items to return")
    similarity_type: str = Field(default="hybrid", description="Type of similarity: 'vector', 'metadata', or 'hybrid'")


class SimilarContentFinder:
    """Finds similar content based on an item."""

    def __init__(self, vector_store):
        self.vector_store = vector_store

    def find(self, item_id: str, n: int = 5, similarity_type: str = "hybrid") -> Dict[str, Any]:
        """
        Find similar content to a given item.

        Args:
            item_id: ID of the reference item
            n: Number of similar items to return
            similarity_type: "vector" (embedding-based), "metadata" (genre/mood), or "hybrid"

        Returns:
            Similar items with similarity scores
        """
        logger.info(f"Finding {n} similar items to {item_id} (type: {similarity_type})")

        try:
            # Get the reference item - try by ID first, then by title search
            ref_metadata = None

            # Try as numeric ID first
            try:
                ref_results = self.vector_store.get_by_ids([str(item_id)])
                if ref_results['metadatas']:
                    ref_metadata = ref_results['metadatas'][0]
            except:
                pass

            # If not found by ID, try searching by title
            if not ref_metadata:
                logger.info(f"ID lookup failed, searching by title: {item_id}")
                search_results = self.vector_store.search(query=item_id, n_results=5)

                if search_results['metadatas'] and len(search_results['metadatas'][0]) > 0:
                    # Find best title match
                    for metadata in search_results['metadatas'][0]:
                        title = metadata.get('title', '').lower()
                        if item_id.lower() in title or title in item_id.lower():
                            ref_metadata = metadata
                            logger.info(f"Found match: {metadata.get('title')}")
                            break

                    # If no exact match, use first result
                    if not ref_metadata:
                        ref_metadata = search_results['metadatas'][0][0]
                        logger.info(f"Using best match: {ref_metadata.get('title')}")

            if not ref_metadata:
                return {
                    "success": False,
                    "error": f"Item '{item_id}' not found. Try a different title or ID.",
                    "items": [],
                    "count": 0
                }

            ref_item = format_content_item(ref_metadata, include_all=True)

            similar_items = []

            if similarity_type in ["vector", "hybrid"]:
                # Vector-based similarity
                query_text = f"{ref_item['title']} {' '.join(ref_item.get('genres', []))} {' '.join(ref_item.get('moods', []))}"
                results = self.vector_store.search(query=query_text, n_results=n + 5)

                if results['ids'] and len(results['ids']) > 0:
                    for i, metadata in enumerate(results['metadatas'][0]):
                        item = format_content_item(metadata, include_all=True)

                        # Skip the reference item itself
                        if str(item['id']) == str(item_id):
                            continue

                        distance = results['distances'][0][i] if results.get('distances') else 0.5
                        item['vector_similarity'] = 1.0 - distance
                        similar_items.append(item)

            if similarity_type in ["metadata", "hybrid"]:
                # Calculate metadata-based similarity
                ref_genres = set([g.lower() for g in ref_item.get('genres', [])])
                ref_moods = set([m.lower() for m in ref_item.get('moods', [])])

                for item in similar_items:
                    item_genres = set([g.lower() for g in item.get('genres', [])])
                    item_moods = set([m.lower() for m in item.get('moods', [])])

                    genre_sim = len(ref_genres & item_genres) / max(len(ref_genres | item_genres), 1)
                    mood_sim = len(ref_moods & item_moods) / max(len(ref_moods | item_moods), 1)

                    item['metadata_similarity'] = (genre_sim + mood_sim) / 2

            # Calculate final similarity score
            for item in similar_items:
                if similarity_type == "vector":
                    item['similarity_score'] = item.get('vector_similarity', 0)
                elif similarity_type == "metadata":
                    item['similarity_score'] = item.get('metadata_similarity', 0)
                else:  # hybrid
                    vec_sim = item.get('vector_similarity', 0)
                    meta_sim = item.get('metadata_similarity', 0)
                    item['similarity_score'] = (vec_sim * 0.6) + (meta_sim * 0.4)

            # Sort and limit
            similar_items.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
            similar_items = similar_items[:n]

            logger.info(f"Found {len(similar_items)} similar items")
            
            # Update context for subsequent tools
            set_items(similar_items)

            return {
                "success": True,
                "reference_item": ref_item,
                "similarity_type": similarity_type,
                "items": similar_items,
                "count": len(similar_items)
            }

        except Exception as e:
            logger.error(f"Similar content finder error: {e}")
            return {"success": False, "error": str(e), "items": [], "count": 0}

    def as_langchain_tool(self) -> StructuredTool:
        def run_tool(item_id: str, n: int = 5, similarity_type: str = "hybrid") -> str:
            try:
                result = self.find(item_id, n, similarity_type)

                if result['success']:
                    ref = result.get('reference_item', {})
                    items = result.get('items', [])
                    
                    # Format output with embedded JSON for parsing
                    output = f"Found {result['count']} items similar to:\n"
                    output += f"'{ref.get('title', 'Unknown')}' ({ref.get('year', 'N/A')})\n\n"
                    output += f"Similarity type: {result['similarity_type']}\n\n"

                    if result['items']:
                        output += "Top similar items:\n"
                        for i, item in enumerate(result['items'][:5], 1):
                            output += f"{i}. {item['title']} ({item['year']}) - {item['type']}\n"
                            output += f"   Similarity: {item.get('similarity_score', 0):.2f}, "
                            output += f"Rating: {item['tmdb_rating']}/10\n"
                    else:
                        output += "No similar items found."

                    # ALWAYS output JSON for machine parsing
                    output += f"\n__ITEMS_JSON__: {json.dumps(result['items'])}"

                    return output
                else:
                    return f"Error: {result.get('error', 'Unknown error')}"

            except Exception as e:
                return f"Error: {str(e)}"

        return StructuredTool.from_function(
            func=run_tool,
            name="similar_content_finder",
            description="Finds content similar to a given item (for 'more like this' queries). Use this when users ask for movies/shows similar to a specific title.",
            args_schema=SimilarContentInput
        )


def create_similar_content_finder_tool(vector_store) -> StructuredTool:
    finder = SimilarContentFinder(vector_store)
    return finder.as_langchain_tool()
