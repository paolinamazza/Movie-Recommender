"""
Request Context Manager

Provides thread-local storage for sharing data (like retrieved items) 
between tools during a single request execution.
"""

import threading
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Thread-local storage
_context = threading.local()

def reset_context():
    """Reset the context for a new request."""
    _context.items = []
    _context.excluded_items = set()
    logger.debug("Context reset")

def set_items(items: List[Dict[str, Any]]):
    """Set the current list of items in context."""
    _context.items = items
    logger.debug(f"Context updated with {len(items)} items")

def get_items() -> List[Dict[str, Any]]:
    """Get the current list of items from context."""
    return getattr(_context, 'items', [])

def add_excluded_items(item_ids: List[str]):
    """Add item IDs to the global exclusion list."""
    if not hasattr(_context, 'excluded_items'):
        _context.excluded_items = set()
    
    _context.excluded_items.update(item_ids)
    logger.debug(f"Added {len(item_ids)} items to exclusion list. Total excluded: {len(_context.excluded_items)}")

def get_excluded_items() -> List[str]:
    """Get the list of globally excluded item IDs."""
    if not hasattr(_context, 'excluded_items'):
        return []
    return list(_context.excluded_items)

def has_items() -> bool:
    """Check if context has items."""
    items = getattr(_context, 'items', [])
    return bool(items and len(items) > 0)
