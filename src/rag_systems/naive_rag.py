"""
Naive RAG System Implementation.

Simple vector similarity search followed by LLM-based ranking and formatting.
No query expansion, no metadata filtering, no advanced reranking.
"""

import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI

from src.utils.vector_store import VectorStoreManager
from src.utils.helpers import parse_metadata_field, format_content_item, deduplicate_results

logger = logging.getLogger(__name__)


class NaiveRAG:
    """
    Naive RAG implementation with basic vector search and LLM formatting.

    Pipeline:
    1. User query -> Vector search (top K results)
    2. Retrieved items -> Format for LLM
    3. LLM generates final recommendations with explanations
    """

    def __init__(self,
                 vector_store: VectorStoreManager,
                 openai_api_key: str,
                 model: str = "gpt-4o",
                 temperature: float = 0.7,
                 top_k: int = 10):
        """
        Initialize Naive RAG system.

        Args:
            vector_store: Initialized VectorStoreManager
            openai_api_key: OpenAI API key
            model: LLM model name
            temperature: LLM temperature
            top_k: Number of results to retrieve
        """
        self.vector_store = vector_store
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model
        self.temperature = temperature
        self.top_k = top_k

        logger.info(f"Initialized NaiveRAG with model={model}, top_k={top_k}")

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant content using vector similarity search.

        Args:
            query: User query string

        Returns:
            List of retrieved content items
        """
        logger.info(f"NaiveRAG retrieving for query: '{query}'")

        # Simple vector search - no metadata filtering
        results = self.vector_store.search(query=query, n_results=self.top_k)

        # Format results
        items = []
        if results['ids'] and len(results['ids']) > 0:
            for metadata in results['metadatas'][0]:
                item = format_content_item(metadata, include_all=True)
                items.append(item)

        logger.info(f"Retrieved {len(items)} items")
        return items

    def generate_response(self, query: str, retrieved_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate final recommendations using LLM.

        Args:
            query: User query
            retrieved_items: List of retrieved items

        Returns:
            Dictionary with recommendations and explanation
        """
        if not retrieved_items:
            return {
                "recommendations": [],
                "explanation": "No matching content found for your query.",
                "method": "naive_rag"
            }

        # Create context from retrieved items
        context = self._format_context(retrieved_items)

        # Create prompt
        prompt = self._create_prompt(query, context)

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

            # Parse LLM response to extract top 5 recommendations
            recommendations = self._parse_llm_response(llm_response, retrieved_items)

            return {
                "recommendations": recommendations[:5],
                "explanation": llm_response,
                "method": "naive_rag",
                "retrieved_count": len(retrieved_items)
            }

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            # Fallback: return top 5 by rating
            sorted_items = sorted(retrieved_items, key=lambda x: x.get('tmdb_rating', 0), reverse=True)
            return {
                "recommendations": sorted_items[:5],
                "explanation": "Showing top-rated content from search results.",
                "method": "naive_rag_fallback",
                "retrieved_count": len(retrieved_items)
            }

    def recommend(self, query: str) -> Dict[str, Any]:
        """
        Complete recommendation pipeline.

        Args:
            query: User query string

        Returns:
            Dictionary with recommendations and metadata
        """
        # Step 1: Retrieve
        retrieved_items = self.retrieve(query)

        # Step 2: Generate response
        result = self.generate_response(query, retrieved_items)

        return result

    def _format_context(self, items: List[Dict[str, Any]]) -> str:
        """Format retrieved items as context for LLM."""
        context_parts = []

        for i, item in enumerate(items, 1):
            part = f"{i}. {item['title']} ({item['year']})\n"
            part += f"   Type: {item['type']}\n"
            part += f"   Rating: {item['tmdb_rating']}/10\n"
            part += f"   Genres: {', '.join(item['genres'])}\n"
            part += f"   Moods: {', '.join(item['moods'])}\n"

            if item['type'] == 'series':
                part += f"   Seasons: {item.get('seasons', 'N/A')}, Episodes: {item.get('episodes', 'N/A')}\n"
            else:
                part += f"   Runtime: {item.get('runtime_minutes', 'N/A')} minutes\n"

            if item.get('cast'):
                part += f"   Cast: {', '.join(item['cast'][:3])}\n"

            context_parts.append(part)

        return "\n".join(context_parts)

    def _get_system_prompt(self) -> str:
        """Get system prompt for LLM."""
        return """You are a movie and TV series recommendation assistant.
Your job is to recommend the most relevant content from the provided list based on the user's query.

Consider:
- User's mood and preferences mentioned in the query
- Genre and mood matches
- Ratings and popularity
- Runtime/length preferences
- Viewing context (solo, family, date night, etc.)

Provide:
1. Top 5 recommendations from the list (by title and year)
2. Brief explanation for each recommendation (1-2 sentences)
3. Overall reasoning for your selections

Be concise and helpful. Focus on matching the user's needs."""

    def _create_prompt(self, query: str, context: str) -> str:
        """Create user prompt for LLM."""
        return f"""User Query: {query}

Available Content:
{context}

Based on the user's query and the available content above, recommend the top 5 most suitable movies/series.
For each recommendation, explain why it matches the user's request.

Format your response as:
RECOMMENDATIONS:
1. [Title] ([Year]) - [Explanation]
2. [Title] ([Year]) - [Explanation]
...

REASONING:
[Overall explanation of your selection strategy]"""

    def _parse_llm_response(self, llm_response: str, retrieved_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Parse LLM response to extract recommended items.

        Args:
            llm_response: LLM's text response
            retrieved_items: Original retrieved items

        Returns:
            Ordered list of recommended items
        """
        # Extract titles mentioned in the response
        lines = llm_response.split('\n')
        recommended = []
        title_to_item = {item['title'].lower(): item for item in retrieved_items}

        for line in lines:
            # Look for numbered recommendations
            if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith('-')):
                # Extract title (text between first space and opening parenthesis)
                for title_key, item in title_to_item.items():
                    if title_key in line.lower():
                        if item not in recommended:
                            recommended.append(item)
                        break

        # If parsing failed, fall back to rating-based sorting
        if not recommended:
            recommended = sorted(retrieved_items, key=lambda x: x.get('tmdb_rating', 0), reverse=True)

        return recommended


def main():
    """Test the Naive RAG system."""
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

    # Initialize Naive RAG
    print("\nInitializing Naive RAG system...")
    naive_rag = NaiveRAG(vector_store=vector_store, openai_api_key=api_key)

    # Test queries
    test_queries = [
        "I want a funny family movie for kids under 2 hours",
        "Suggest a scary horror movie for tonight",
        "Looking for an inspiring drama about overcoming challenges"
    ]

    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print('='*80)

        result = naive_rag.recommend(query)

        print(f"\nMethod: {result['method']}")
        print(f"Retrieved: {result['retrieved_count']} items")
        print(f"\nTop 5 Recommendations:")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"{i}. {rec['title']} ({rec['year']}) - {rec['type']}")
            print(f"   Rating: {rec['tmdb_rating']}/10, Genres: {', '.join(rec['genres'])}")

        print(f"\nExplanation:\n{result['explanation'][:300]}...")


if __name__ == "__main__":
    main()
