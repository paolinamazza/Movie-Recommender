"""
Advanced RAG System Implementation.

Features:
- HyDE (Hypothetical Document Embeddings) for query expansion
- Metadata filtering based on extracted constraints
- Hybrid search (vector + keyword)
- Reranking by rating, popularity, and relevance
"""

import logging
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI

from src.utils.vector_store import VectorStoreManager
from src.utils.helpers import (
    parse_metadata_field,
    format_content_item,
    deduplicate_results,
    calculate_constraint_satisfaction
)

logger = logging.getLogger(__name__)


class AdvancedRAG:
    """
    Advanced RAG with query expansion, filtering, and reranking.

    Pipeline:
    1. Query analysis -> Extract constraints (mood, type, rating, etc.)
    2. HyDE -> Generate hypothetical ideal document
    3. Hybrid search -> Vector search with metadata filters
    4. Reranking -> Score by rating, popularity, constraint match
    5. LLM generation -> Format final recommendations
    """

    def __init__(self,
                 vector_store: VectorStoreManager,
                 openai_api_key: str,
                 model: str = "gpt-4o",
                 temperature: float = 0.7,
                 top_k: int = 20,
                 final_k: int = 5):
        """
        Initialize Advanced RAG system.

        Args:
            vector_store: Initialized VectorStoreManager
            openai_api_key: OpenAI API key
            model: LLM model name
            temperature: LLM temperature
            top_k: Number of results to retrieve before reranking
            final_k: Number of final recommendations
        """
        self.vector_store = vector_store
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.final_k = final_k

        logger.info(f"Initialized AdvancedRAG with model={model}, top_k={top_k}")

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Extract constraints and intent from user query using LLM.

        Args:
            query: User query string

        Returns:
            Dictionary with extracted constraints
        """
        prompt = f"""Analyze this movie/series recommendation query and extract key constraints.

Query: {query}

Extract the following if mentioned (return null if not specified):
- mood: emotional mood (happy, sad, excited, scared, relaxed, curious, inspired, stressed, nostalgic, bored)
- type: "movie" or "series" or null
- context: viewing situation (solo_watch, family_watch, date_night, background_watch, binge_worthy)
- max_hours: maximum runtime in hours (as a number)
- min_rating: minimum rating 1-10 (as a number)
- genre: specific genre if mentioned
- other: any other preferences

Respond in JSON format:
{{
  "mood": "...",
  "type": "...",
  "context": "...",
  "max_hours": ...,
  "min_rating": ...,
  "genre": "...",
  "other": "..."
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a query analysis assistant. Extract constraints from user queries and respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )

            content = response.choices[0].message.content.strip()

            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                constraints = json.loads(json_match.group())
                # Clean up null values
                constraints = {k: v for k, v in constraints.items() if v is not None and v != "null"}
                logger.info(f"Extracted constraints: {constraints}")
                return constraints
            else:
                logger.warning("Failed to extract JSON from query analysis")
                return {}

        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            return {}

    def generate_hyde_document(self, query: str, constraints: Dict[str, Any]) -> str:
        """
        Generate hypothetical ideal document using HyDE technique.

        Args:
            query: User query
            constraints: Extracted constraints

        Returns:
            Hypothetical document text
        """
        prompt = f"""Given this user request for a movie/series recommendation, write a brief description of an ideal movie or series that would perfectly match their needs.

User Request: {query}

Constraints: {json.dumps(constraints)}

Write a 2-3 sentence description of the perfect content that would satisfy this request. Include genre, mood, themes, and why it would be perfect for their needs.

Ideal content description:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a recommendation assistant. Generate concise descriptions of ideal content."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )

            hyde_doc = response.choices[0].message.content.strip()
            logger.info(f"Generated HyDE document: {hyde_doc[:100]}...")
            return hyde_doc

        except Exception as e:
            logger.error(f"HyDE generation failed: {e}")
            return query  # Fallback to original query

    def retrieve(self, query: str, hyde_doc: str, constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Retrieve relevant content using hybrid search with filters.

        Args:
            query: Original user query
            hyde_doc: HyDE document for enhanced search
            constraints: Extracted constraints for filtering

        Returns:
            List of retrieved content items
        """
        # Build metadata filter from constraints
        where_filter = self._build_metadata_filter(constraints)

        # Search with HyDE document (better semantic match)
        search_text = f"{query} {hyde_doc}"

        logger.info(f"Retrieving with filter: {where_filter}")
        results = self.vector_store.search(
            query=search_text,
            n_results=self.top_k,
            where=where_filter if where_filter else None
        )

        # Format results
        items = []
        if results['ids'] and len(results['ids']) > 0:
            for i, metadata in enumerate(results['metadatas'][0]):
                item = format_content_item(metadata, include_all=True)
                # Store relevance score (distance)
                item['relevance_score'] = 1.0 - results['distances'][0][i] if results.get('distances') else 0.5
                items.append(item)

        logger.info(f"Retrieved {len(items)} items")
        return items

    def rerank(self, items: List[Dict[str, Any]], constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Rerank items based on multiple factors.

        Scoring:
        - Relevance score: 0.3 weight
        - Rating (TMDB): 0.3 weight (normalized to 0-1)
        - Popularity: 0.2 weight (normalized)
        - Constraint satisfaction: 0.2 weight

        Args:
            items: Retrieved items
            constraints: Extracted constraints

        Returns:
            Reranked items
        """
        if not items:
            return items

        # Normalize popularity
        max_pop = max([item.get('popularity', 0) for item in items] + [1])

        for item in items:
            score = 0.0

            # Relevance score (from vector distance)
            score += 0.3 * item.get('relevance_score', 0.5)

            # Rating score (normalize 0-10 to 0-1)
            rating = item.get('tmdb_rating', 0)
            score += 0.3 * (rating / 10.0)

            # Popularity score (normalized)
            popularity = item.get('popularity', 0)
            score += 0.2 * (popularity / max_pop)

            # Constraint satisfaction
            if constraints:
                satisfaction = calculate_constraint_satisfaction(item, constraints)
                constraint_score = sum(satisfaction.values()) / len(satisfaction) if satisfaction else 0.5
                score += 0.2 * constraint_score

            item['rerank_score'] = score

        # Sort by rerank score
        reranked = sorted(items, key=lambda x: x.get('rerank_score', 0), reverse=True)
        logger.info(f"Reranked {len(reranked)} items")

        return reranked

    def generate_response(self, query: str, ranked_items: List[Dict[str, Any]], constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate final recommendations using LLM.

        Args:
            query: User query
            ranked_items: Reranked items
            constraints: Extracted constraints

        Returns:
            Dictionary with recommendations and explanation
        """
        if not ranked_items:
            return {
                "recommendations": [],
                "explanation": "No matching content found for your query.",
                "method": "advanced_rag",
                "constraints": constraints
            }

        # Take top results after reranking
        top_items = ranked_items[:self.final_k * 2]  # Get 2x for LLM to choose from

        # Create context
        context = self._format_context(top_items)

        # Create prompt
        prompt = self._create_prompt(query, context, constraints)

        # Call LLM
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=1000
            )

            llm_response = response.choices[0].message.content

            # Parse response to get final recommendations
            recommendations = self._parse_llm_response(llm_response, ranked_items)

            return {
                "recommendations": recommendations[:self.final_k],
                "explanation": llm_response,
                "method": "advanced_rag",
                "constraints": constraints,
                "retrieved_count": len(ranked_items)
            }

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return {
                "recommendations": ranked_items[:self.final_k],
                "explanation": "Showing top reranked results.",
                "method": "advanced_rag_fallback",
                "constraints": constraints,
                "retrieved_count": len(ranked_items)
            }

    def recommend(self, query: str) -> Dict[str, Any]:
        """
        Complete Advanced RAG pipeline.

        Args:
            query: User query string

        Returns:
            Dictionary with recommendations and metadata
        """
        # Step 1: Analyze query
        constraints = self.analyze_query(query)

        # Step 2: Generate HyDE document
        hyde_doc = self.generate_hyde_document(query, constraints)

        # Step 3: Retrieve with filters
        retrieved_items = self.retrieve(query, hyde_doc, constraints)

        # Step 4: Rerank
        ranked_items = self.rerank(retrieved_items, constraints)

        # Step 5: Generate response
        result = self.generate_response(query, ranked_items, constraints)

        return result

    def _build_metadata_filter(self, constraints: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build ChromaDB metadata filter from constraints."""
        where_filter = {}

        # Type filter
        if constraints.get('type') in ['movie', 'series']:
            where_filter['type'] = constraints['type']

        # Note: Other filters (mood, genre) are handled in reranking
        # because ChromaDB doesn't support list membership queries easily

        return where_filter if where_filter else None

    def _format_context(self, items: List[Dict[str, Any]]) -> str:
        """Format items as context for LLM."""
        context_parts = []

        for i, item in enumerate(items, 1):
            part = f"{i}. {item['title']} ({item['year']})\n"
            part += f"   Type: {item['type']}\n"
            part += f"   Rating: {item['tmdb_rating']}/10\n"
            part += f"   Genres: {', '.join(item['genres'])}\n"
            part += f"   Moods: {', '.join(item['moods'])}\n"
            part += f"   Relevance Score: {item.get('rerank_score', 0):.2f}\n"

            context_parts.append(part)

        return "\n".join(context_parts)

    def _get_system_prompt(self) -> str:
        """Get system prompt for LLM."""
        return """You are an expert movie and TV series recommendation assistant.
You have access to a pre-filtered and reranked list of content that matches the user's query.

Your job is to select the top 5 recommendations and explain why they're perfect for the user's needs.

Consider the relevance scores, ratings, and how well each item matches the user's stated preferences."""

    def _create_prompt(self, query: str, context: str, constraints: Dict[str, Any]) -> str:
        """Create user prompt for LLM."""
        return f"""User Query: {query}

Detected Preferences: {json.dumps(constraints, indent=2)}

Top Candidates (already filtered and reranked):
{context}

Select the top 5 recommendations from the list above that best match the user's query and preferences.
For each, explain why it's a great match.

Format:
RECOMMENDATIONS:
1. [Title] ([Year]) - [Explanation]
2. [Title] ([Year]) - [Explanation]
...

REASONING:
[Overall explanation]"""

    def _parse_llm_response(self, llm_response: str, ranked_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse LLM response to extract recommended items."""
        lines = llm_response.split('\n')
        recommended = []
        title_to_item = {item['title'].lower(): item for item in ranked_items}

        for line in lines:
            if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith('-')):
                for title_key, item in title_to_item.items():
                    if title_key in line.lower():
                        if item not in recommended:
                            recommended.append(item)
                        break

        # Fallback to rerank order
        if not recommended:
            recommended = ranked_items

        return recommended


def main():
    """Test the Advanced RAG system."""
    import os
    from dotenv import load_dotenv
    from src.utils.vector_store import initialize_vector_store

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("Error: OPENAI_API_KEY not found")
        return

    # Initialize vector store
    logging.basicConfig(level=logging.INFO)
    print("Initializing vector store...")
    vector_store = initialize_vector_store(api_key, reset=False)

    # Initialize Advanced RAG
    print("\nInitializing Advanced RAG system...")
    advanced_rag = AdvancedRAG(vector_store=vector_store, openai_api_key=api_key)

    # Test queries
    test_queries = [
        "I want a funny family movie for kids under 2 hours",
        "Suggest a scary horror movie for tonight",
        "Looking for an inspiring biographical series I can binge watch"
    ]

    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print('='*80)

        result = advanced_rag.recommend(query)

        print(f"\nMethod: {result['method']}")
        print(f"Constraints: {json.dumps(result.get('constraints', {}), indent=2)}")
        print(f"Retrieved: {result.get('retrieved_count', 0)} items")
        print(f"\nTop 5 Recommendations:")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"{i}. {rec['title']} ({rec['year']}) - {rec['type']}")
            print(f"   Rating: {rec['tmdb_rating']}/10, Rerank Score: {rec.get('rerank_score', 0):.2f}")

        print(f"\nExplanation:\n{result['explanation'][:300]}...")


if __name__ == "__main__":
    main()
