"""
Agentic RAG System Implementation.

Uses a LangChain agent (currently limited to the mood_content_finder tool) for recommendation queries.
"""

import logging
import json
from typing import List, Dict, Any, Optional
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage, HumanMessage

from src.utils.vector_store import VectorStoreManager
from src.tools.mood_content_finder import create_mood_content_finder_tool
from src.tools.context_filter import create_context_filter_tool
from src.tools.runtime_matcher import create_runtime_matcher_tool
from src.tools.viewing_history_checker import create_viewing_history_checker_tool
from src.tools.similar_content_finder import create_similar_content_finder_tool
from src.tools.constraint_relaxer import create_constraint_relaxer_tool
from src.tools.group_preference_reconciler import create_group_preference_reconciler_tool
from src.tools.general_search_tool import create_general_search_tool
from src.utils.request_context import reset_context

logger = logging.getLogger(__name__)


class AgenticRAG:
    """
    Agentic RAG with LangChain agent and specialized tools.

    Pipeline:
    1. Agent receives user query
    2. Agent reasons about which tools to use
    3. Agent executes tools in sequence (mood finder -> context filter -> runtime matcher, etc.)
    4. Agent synthesizes final recommendations with explanations
    """

    def __init__(self,
                 vector_store: VectorStoreManager,
                 openai_api_key: str,
                 model: str = "gpt-4o",
                 temperature: float = 0.7,
                 max_iterations: int = 10,
                 verbose: bool = True):
        """
        Initialize Agentic RAG system.

        Args:
            vector_store: Initialized VectorStoreManager
            openai_api_key: OpenAI API key
            model: LLM model name
            temperature: LLM temperature
            max_iterations: Maximum agent iterations
            verbose: Enable verbose logging
        """
        self.vector_store = vector_store
        self.model = model
        self.temperature = temperature
        self.verbose = verbose

        # Initialize LLM
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model=model,
            temperature=temperature
        )

        # Initialize tools
        self.tools = self._initialize_tools()

        # Create agent
        self.agent = self._create_agent()

        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            max_iterations=max_iterations,
            verbose=verbose,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )

        logger.info(f"Initialized AgenticRAG with {len(self.tools)} tools, model={model}")

    def _initialize_tools(self) -> List:
        """Initialize the full suite of agent tools."""
        tools = [
            create_mood_content_finder_tool(self.vector_store),
            create_context_filter_tool(self.vector_store),
            create_runtime_matcher_tool(self.vector_store),
            create_viewing_history_checker_tool(self.vector_store),
            create_similar_content_finder_tool(self.vector_store),
            create_constraint_relaxer_tool(self.vector_store),
            create_group_preference_reconciler_tool(self.vector_store),
            create_general_search_tool(self.vector_store),
        ]

        logger.info(f"Initialized {len(tools)} tools: {[t.name for t in tools]}")
        return tools

    def _create_agent(self):
        """Create the LangChain OpenAI Functions agent."""
        system_prompt = """You are CineMatch AI, an intelligent movie recommendation assistant. Use these specialized tools:

**Search Tools** (find content):
- mood_content_finder(mood, content_type?, min_rating?) - Find by mood/emotion
- similar_content_finder(item_id, n?, similarity_type?) - Find similar to a title
- general_search_tool(query, n?) - Semantic search for topics, plots, or genres (e.g. "space movies", "90s action")

**Filter Tools** (refine results - can work standalone OR accept items from previous tools):
- context_filter(context, items?) - Filter by viewing context (family/date/solo/background/binge)
- runtime_matcher(available_hours, items?, session_type?) - Match to available time
- viewing_history_checker(user_history, items?, avoid_similar?) - Avoid watched content

**Special Tools**:
- constraint_relaxer(requirements, results_count) - Suggest relaxations when no results
- group_preference_reconciler(preferences_list) - Merge multiple users' preferences

**How to use**:
1. Start with a search tool (mood_content_finder, similar_content_finder, or general_search_tool)
2. Optionally apply filters - they'll work on their own if no items provided
3. Return up to 5 recommendations with title, year, type, rating, and why they match

Map moods: funny→happy, scary→scared, cozy→relaxed, intense→excited, etc.
Be conversational and helpful."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )

        return agent

    def recommend(self, query: str, chat_history: Optional[List] = None) -> Dict[str, Any]:
        """
        Generate recommendations using the agent.

        Args:
            query: User query string
            chat_history: Optional conversation history

        Returns:
            Dictionary with recommendations, reasoning, and tool trace
        """
        logger.info(f"Agentic RAG processing query: '{query}'")
        reset_context()

        try:
            # Execute agent
            result = self.agent_executor.invoke({
                "input": query,
                "chat_history": chat_history or []
            })

            # Extract output and intermediate steps
            output = result.get("output", "")
            intermediate_steps = result.get("intermediate_steps", [])

            # Parse recommendations from output
            recommendations = self._parse_recommendations(output, intermediate_steps)

            # Build tool trace for transparency
            tool_trace = self._build_tool_trace(intermediate_steps)

            return {
                "recommendations": recommendations,
                "explanation": output,
                "method": "agentic_rag",
                "tool_trace": tool_trace,
                "tool_calls": len(intermediate_steps)
            }

        except Exception as e:
            logger.error(f"Agentic RAG error: {e}")
            return {
                "recommendations": [],
                "explanation": f"Error processing query: {str(e)}",
                "method": "agentic_rag_error",
                "tool_trace": [],
                "tool_calls": 0
            }

    def _parse_recommendations(self, output: str, intermediate_steps: List) -> List[Dict[str, Any]]:
        """
        Parse recommendations from agent output and tool results.

        Args:
            output: Agent's final output text
            intermediate_steps: List of (AgentAction, result) tuples

        Returns:
            List of recommendation dictionaries
        """
        recommendations = []

        # Try to extract recommendations from tool observation strings
        # Iterate in REVERSE to get the final filtered results first
        logger.info(f"Parsing recommendations from {len(intermediate_steps)} steps (reverse order)")
        
        for i, step in enumerate(reversed(intermediate_steps)):
            action, observation = step
            tool_name = action.tool if hasattr(action, 'tool') else 'unknown'
            logger.info(f"Checking step {len(intermediate_steps)-i}: {tool_name}")

            # Look for tools that return items
            if tool_name in ['mood_content_finder', 'context_filter', 'runtime_matcher',
                             'viewing_history_checker', 'similar_content_finder', 'general_search_tool']:
                try:
                    # Try to extract JSON payload from observation string
                    obs_str = str(observation)

                    # Look for __ITEMS_JSON__ marker
                    if '__ITEMS_JSON__:' in obs_str:
                        json_start = obs_str.find('__ITEMS_JSON__:') + len('__ITEMS_JSON__:')
                        json_str = obs_str[json_start:].strip()
                        logger.info(f"Found __ITEMS_JSON__ in {tool_name}, parsing: {json_str[:100]}")
                        items = json.loads(json_str)

                        if isinstance(items, list):
                            logger.info(f"Parsed {len(items)} items from {tool_name}")
                            # Found the final set of items!
                            for item in items[:5]:
                                if isinstance(item, dict):
                                    recommendations.append(item)
                            
                            logger.info(f"Added {len(recommendations)} items to recommendations, breaking loop")
                            # We found the most refined results, stop looking
                            break
                        else:
                            logger.warning(f"Found __ITEMS_JSON__ in {tool_name} but it was not a list")
                    else:
                        logger.debug(f"No __ITEMS_JSON__ marker in {tool_name} output")

                except Exception as e:
                    logger.error(f"Could not parse items from tool {tool_name}: {e}")

        logger.info(f"Final extracted recommendations: {len(recommendations)}")
        return recommendations[:5]

    def _build_tool_trace(self, intermediate_steps: List) -> List[Dict[str, str]]:
        """
        Build a human-readable trace of tool executions.

        Args:
            intermediate_steps: List of (AgentAction, result) tuples

        Returns:
            List of tool execution summaries
        """
        trace = []

        for i, step in enumerate(intermediate_steps, 1):
            action, observation = step
            tool_name = action.tool if hasattr(action, 'tool') else 'unknown'
            tool_input = action.tool_input if hasattr(action, 'tool_input') else ''

            # Truncate observation if too long
            obs_str = str(observation)
            if len(obs_str) > 200:
                obs_str = obs_str[:200] + "..."

            # Determine success based on observation
            # Consider successful if observation doesn't contain "Error:" or "error"
            obs_lower = str(observation).lower()
            success = not ('error:' in obs_lower or observation == '')

            trace.append({
                "step": i,
                "tool": tool_name,
                "input": str(tool_input)[:100],
                "output_summary": obs_str,
                "success": success,
                "result_summary": obs_str  # Add result_summary for frontend compatibility
            })

        return trace


def main():
    """Test the Agentic RAG system."""
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

    # Initialize Agentic RAG
    print("\nInitializing Agentic RAG system...")
    agentic_rag = AgenticRAG(
        vector_store=vector_store,
        openai_api_key=api_key,
        verbose=True
    )

    # Test queries
    test_queries = [
        "I want a happy family movie under 2 hours that we haven't watched before. We recently saw Toy Story and Frozen.",
        "Find me a scary horror movie for a solo watch tonight, something really intense",
        "Looking for an inspiring biographical series I can binge watch over the weekend, at least 2 seasons"
    ]

    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print('='*80)

        result = agentic_rag.recommend(query)

        print(f"\nMethod: {result['method']}")
        print(f"Tool Calls: {result['tool_calls']}")

        print(f"\nTool Execution Trace:")
        for trace_item in result.get('tool_trace', []):
            print(f"  {trace_item['step']}. {trace_item['tool']}")
            print(f"     Input: {trace_item['input']}")
            print(f"     Output: {trace_item['output_summary']}\n")

        print(f"\nAgent Response:")
        print(result['explanation'])

        if result['recommendations']:
            print(f"\nExtracted Recommendations:")
            for i, rec in enumerate(result['recommendations'], 1):
                print(f"{i}. {rec.get('title', 'Unknown')} ({rec.get('year', 'N/A')})")


if __name__ == "__main__":
    main()
