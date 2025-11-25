"""
Helper functions for RAG systems.
"""

import json
from typing import List, Dict, Any


def parse_metadata_field(metadata: Dict[str, Any], field: str) -> Any:
    """
    Parse JSON-serialized metadata fields from ChromaDB.

    Args:
        metadata: Metadata dictionary from ChromaDB
        field: Field name to parse

    Returns:
        Parsed value (list or dict) or empty list/dict if parsing fails
    """
    value = metadata.get(field, '')
    if not value:
        return [] if field in ['genres', 'moods', 'keywords', 'cast', 'crew'] else {}

    try:
        return json.loads(value) if isinstance(value, str) else value
    except (json.JSONDecodeError, TypeError):
        return [] if field in ['genres', 'moods', 'keywords', 'cast', 'crew'] else {}


def format_content_item(metadata: Dict[str, Any], include_all: bool = False) -> Dict[str, Any]:
    """
    Format a content item from ChromaDB metadata.

    Args:
        metadata: Metadata dictionary from ChromaDB
        include_all: If True, include all fields; if False, only essential fields

    Returns:
        Formatted content dictionary
    """
    item = {
        "id": metadata.get("content_id"),
        "type": metadata.get("type"),
        "title": metadata.get("title"),
        "year": metadata.get("year"),
        "tmdb_rating": metadata.get("tmdb_rating"),
        "genres": parse_metadata_field(metadata, "genres"),
        "moods": parse_metadata_field(metadata, "moods"),
    }

    if include_all:
        item.update({
            "vote_count": metadata.get("vote_count"),
            "runtime_minutes": metadata.get("runtime_minutes"),
            "total_hours": metadata.get("total_hours"),
            "language": metadata.get("language"),
            "content_rating": metadata.get("content_rating"),
            "keywords": parse_metadata_field(metadata, "keywords"),
            "context_suitability": parse_metadata_field(metadata, "context_suitability"),
            "cast": parse_metadata_field(metadata, "cast"),
            "crew": parse_metadata_field(metadata, "crew"),
        })

        if metadata.get("type") == "series":
            item["seasons"] = metadata.get("seasons")
            item["episodes"] = metadata.get("episodes")

    return item


def calculate_constraint_satisfaction(
    item: Dict[str, Any],
    constraints: Dict[str, Any]
) -> Dict[str, bool]:
    """
    Check which constraints are satisfied by an item.

    Args:
        item: Content item dictionary
        constraints: Dictionary of constraints to check

    Returns:
        Dictionary mapping constraint names to satisfaction status
    """
    satisfaction = {}

    # Mood constraint
    if "mood" in constraints:
        target_mood = constraints["mood"].lower()
        item_moods = [m.lower() for m in item.get("moods", [])]
        satisfaction["mood"] = target_mood in item_moods

    # Context constraint
    if "context" in constraints:
        context = constraints["context"]
        suitability = item.get("context_suitability", {})
        if isinstance(suitability, str):
            suitability = json.loads(suitability)
        satisfaction["context"] = suitability.get(context, 0) >= 0.6

    # Runtime constraint
    if "max_hours" in constraints:
        max_hours = constraints["max_hours"]
        item_hours = item.get("total_hours", 0)
        satisfaction["runtime"] = item_hours <= max_hours

    # Minimum rating constraint
    if "min_rating" in constraints:
        min_rating = constraints["min_rating"]
        item_rating = item.get("tmdb_rating", 0)
        satisfaction["rating"] = item_rating >= min_rating

    # Content type constraint
    if "type" in constraints:
        content_type = constraints["type"]
        satisfaction["type"] = item.get("type") == content_type

    # Genre constraint
    if "genre" in constraints:
        target_genre = constraints["genre"].lower()
        item_genres = [g.lower() for g in item.get("genres", [])]
        satisfaction["genre"] = target_genre in item_genres

    return satisfaction


def calculate_csr(items: List[Dict[str, Any]], constraints: Dict[str, Any]) -> float:
    """
    Calculate Constraint Satisfaction Rate.

    Args:
        items: List of recommended items
        constraints: Dictionary of constraints

    Returns:
        CSR score (0-1)
    """
    if not items or not constraints:
        return 0.0

    total_constraints = len(constraints)
    satisfied_count = 0

    for item in items:
        satisfaction = calculate_constraint_satisfaction(item, constraints)
        satisfied_count += sum(satisfaction.values())

    max_possible = total_constraints * len(items)
    return satisfied_count / max_possible if max_possible > 0 else 0.0


def format_recommendation_text(item: Dict[str, Any], rank: int) -> str:
    """
    Format a single recommendation as readable text.

    Args:
        item: Content item dictionary
        rank: Recommendation rank (1-based)

    Returns:
        Formatted text string
    """
    title = item.get("title", "Unknown")
    year = item.get("year", "N/A")
    content_type = item.get("type", "content").capitalize()
    rating = item.get("tmdb_rating", 0)
    genres = item.get("genres", [])
    runtime = item.get("runtime_minutes", 0)

    text = f"{rank}. {title} ({year})\n"
    text += f"   Type: {content_type} | Rating: {rating:.1f}/10\n"
    text += f"   Genres: {', '.join(genres)}\n"

    if item.get("type") == "series":
        seasons = item.get("seasons", 0)
        episodes = item.get("episodes", 0)
        text += f"   Seasons: {seasons} | Episodes: {episodes}\n"
    else:
        text += f"   Runtime: {runtime} min\n"

    return text


def deduplicate_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate items from results based on ID.

    Args:
        results: List of content items

    Returns:
        Deduplicated list
    """
    seen_ids = set()
    unique_results = []

    for item in results:
        item_id = item.get("id")
        if item_id not in seen_ids:
            seen_ids.add(item_id)
            unique_results.append(item)

    return unique_results
