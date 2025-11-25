"""
AgenticRAGV2: Enhanced Agentic RAG with Planning, Reflection, and Memory.

This implementation integrates:
1. AgenticRAG - Base agent with specialized tools
2. Planner - Multi-step execution planning
3. Reflector - Quality evaluation and improvement suggestions
4. ConversationMemory - Conversation tracking and preference learning

Provides a complete pipeline: Plan -> Execute -> Reflect -> Explain
"""

import logging
import json
from typing import List, Dict, Any, Optional
from langchain.schema import HumanMessage, AIMessage

from src.rag_systems.agentic_rag import AgenticRAG
from src.planning.planner import Planner, ExecutionPlan, PlanStep
from src.planning.reflector import Reflector, Reflection
from src.planning.memory import ConversationMemory
from src.utils.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class AgenticRAGV2:
    """
    Enhanced Agentic RAG with planning, reflection, and memory.

    Pipeline:
    1. Plan: Analyze query and create execution strategy
    2. Execute: Run agent with planned steps
    3. Reflect: Evaluate quality and suggest improvements
    4. Explain: Generate comprehensive explanation of process and results

    Returns detailed process information including plans, traces, reflections, and explanations.
    """

    def __init__(self,
                 vector_store: VectorStoreManager,
                 openai_api_key: str,
                 model: str = "gpt-4o-mini",
                 temperature: float = 0.7,
                 max_iterations: int = 10,
                 enable_planning: bool = True,
                 enable_reflection: bool = True,
                 enable_memory: bool = True,
                 verbose: bool = True):
        """
        Initialize AgenticRAGV2 system.

        Args:
            vector_store: Initialized VectorStoreManager
            openai_api_key: OpenAI API key
            model: LLM model name
            temperature: LLM temperature
            max_iterations: Maximum agent iterations
            enable_planning: Enable multi-step planning
            enable_reflection: Enable quality reflection
            enable_memory: Enable conversation memory
            verbose: Enable verbose logging
        """
        self.vector_store = vector_store
        self.model = model
        self.temperature = temperature
        self.verbose = verbose
        self.openai_api_key = openai_api_key

        # Feature flags
        self.enable_planning = enable_planning
        self.enable_reflection = enable_reflection
        self.enable_memory = enable_memory

        # Initialize base AgenticRAG
        self.agentic_rag = AgenticRAG(
            vector_store=vector_store,
            openai_api_key=openai_api_key,
            model=model,
            temperature=temperature,
            max_iterations=max_iterations,
            verbose=verbose
        )

        # Initialize planning components
        if enable_planning:
            self.planner = Planner(api_key=openai_api_key)
            logger.info("Planner initialized")
        else:
            self.planner = None

        # Initialize reflection
        if enable_reflection:
            self.reflector = Reflector(api_key=openai_api_key)
            logger.info("Reflector initialized")
        else:
            self.reflector = None

        # Initialize conversation memory
        if enable_memory:
            self.memory = ConversationMemory()
            logger.info("ConversationMemory initialized")
        else:
            self.memory = None

        logger.info(
            f"Initialized AgenticRAGV2 with model={model}, "
            f"planning={enable_planning}, reflection={enable_reflection}, memory={enable_memory}"
        )

    def recommend(self, query: str, chat_history: Optional[List] = None) -> Dict[str, Any]:
        """
        Generate recommendations with planning, execution, reflection, and explanation.

        Args:
            query: User query string
            chat_history: Optional conversation history (LangChain format)

        Returns:
            Dictionary with:
            - recommendations: List of movie/show dictionaries
            - explanation: Natural language explanation
            - method: "agentic_rag_v2"
            - tool_calls: Number of tool calls made
            - planning_trace: Details of the execution plan
            - execution_trace: Step-by-step execution details
            - reflection_results: Quality evaluation results
            - detailed_explanation: Comprehensive process explanation
        """
        logger.info(f"AgenticRAGV2 processing query: '{query}'")

        try:
            # Stage 1: Planning
            plan = None
            planning_trace = {}

            if self.enable_planning and self.planner:
                plan = self._create_execution_plan(query, chat_history)
                planning_trace = self._serialize_plan(plan)
                logger.info(f"Created plan with strategy: {plan.strategy}, steps: {len(plan.steps)}")

            # Stage 2: Execution
            execution_result = self._execute_with_agent(query, chat_history, plan)

            # Stage 3: Reflection
            reflection = None
            reflection_results = {}

            if self.enable_reflection and self.reflector:
                reflection = self._reflect_on_results(
                    query=query,
                    results=execution_result.get('recommendations', []),
                    plan=plan
                )
                reflection_results = self._serialize_reflection(reflection)
                logger.info(f"Reflection quality score: {reflection.quality_score:.2f}")

            # Stage 4: Generate Detailed Explanation
            detailed_explanation = self._generate_detailed_explanation(
                query=query,
                plan=plan,
                execution_result=execution_result,
                reflection=reflection
            )

            # Stage 5: Update Memory
            if self.enable_memory and self.memory:
                self._update_memory(
                    query=query,
                    recommendations=execution_result.get('recommendations', []),
                    plan=plan
                )

            # Build comprehensive result
            result = {
                "recommendations": execution_result.get('recommendations', []),
                "explanation": execution_result.get('explanation', ''),
                "method": "agentic_rag_v2",
                "tool_calls": execution_result.get('tool_calls', 0),
                "planning_trace": planning_trace,
                "execution_trace": execution_result.get('tool_trace', []),
                "reflection_results": reflection_results,
                "detailed_explanation": detailed_explanation
            }

            logger.info(
                f"AgenticRAGV2 completed: {len(result['recommendations'])} recommendations, "
                f"{result['tool_calls']} tool calls"
            )

            return result

        except Exception as e:
            logger.error(f"AgenticRAGV2 error: {e}", exc_info=True)
            return {
                "recommendations": [],
                "explanation": f"Error processing query: {str(e)}",
                "method": "agentic_rag_v2_error",
                "tool_calls": 0,
                "planning_trace": {},
                "execution_trace": [],
                "reflection_results": {},
                "detailed_explanation": f"An error occurred during processing: {str(e)}"
            }

    def _create_execution_plan(self, query: str, chat_history: Optional[List] = None) -> ExecutionPlan:
        """Create an execution plan using the Planner."""
        try:
            # Get conversation context from memory if available
            conversation_context = None
            if self.enable_memory and self.memory:
                conversation_context = self.memory.get_recent_context(n=3)

            plan = self.planner.create_plan(
                query=query,
                conversation_context=conversation_context
            )

            return plan

        except Exception as e:
            logger.error(f"Planning error: {e}")
            # Return a simple fallback plan
            return self.planner._create_fallback_plan(query)

    def _execute_with_agent(self,
                           query: str,
                           chat_history: Optional[List] = None,
                           plan: Optional[ExecutionPlan] = None) -> Dict[str, Any]:
        """Execute the query using the base AgenticRAG agent."""
        try:
            # Optionally enhance the query with planning hints
            enhanced_query = query

            if plan and len(plan.steps) > 0:
                # Add planning guidance to the query
                planning_hint = f"\n\n[System Planning Guidance: Consider using {plan.steps[0].tool} first based on query analysis]"
                enhanced_query = query + planning_hint

            # Execute with base AgenticRAG
            result = self.agentic_rag.recommend(
                query=enhanced_query,
                chat_history=chat_history
            )

            return result

        except Exception as e:
            logger.error(f"Execution error: {e}")
            return {
                "recommendations": [],
                "explanation": f"Execution error: {str(e)}",
                "tool_trace": [],
                "tool_calls": 0
            }

    def _reflect_on_results(self,
                           query: str,
                           results: List[Dict],
                           plan: Optional[ExecutionPlan] = None) -> Reflection:
        """Reflect on the quality of results."""
        try:
            # Use the first plan step as context if available
            plan_step = plan.steps[0] if plan and len(plan.steps) > 0 else None

            reflection = self.reflector.evaluate(
                query=query,
                results=results,
                plan_step=plan_step
            )

            return reflection

        except Exception as e:
            logger.error(f"Reflection error: {e}")
            # Return a default reflection
            return Reflection(
                success=len(results) > 0,
                quality_score=0.5,
                issues=[],
                suggestions=[],
                should_retry=False,
                adjusted_parameters={}
            )

    def _generate_detailed_explanation(self,
                                      query: str,
                                      plan: Optional[ExecutionPlan],
                                      execution_result: Dict[str, Any],
                                      reflection: Optional[Reflection]) -> str:
        """Generate a comprehensive explanation of the entire process."""
        try:
            explanation_parts = []

            # Add query understanding
            explanation_parts.append(f"Query Analysis: '{query}'")

            # Add planning details
            if plan:
                explanation_parts.append(
                    f"\nExecution Strategy: {plan.strategy.replace('_', ' ').title()}"
                )
                explanation_parts.append(
                    f"Planned {len(plan.steps)} step(s) with estimated quality: {plan.estimated_quality:.2f}"
                )

                if len(plan.steps) > 0:
                    explanation_parts.append("\nPlanned Steps:")
                    for step in plan.steps:
                        explanation_parts.append(
                            f"  {step.step_number}. {step.tool} - {step.reasoning}"
                        )

            # Add execution summary
            tool_calls = execution_result.get('tool_calls', 0)
            explanation_parts.append(f"\nExecution: Used {tool_calls} tool call(s)")

            tool_trace = execution_result.get('tool_trace', [])
            if tool_trace:
                explanation_parts.append("Tools Used:")
                for trace_item in tool_trace:
                    explanation_parts.append(f"  - {trace_item['tool']}")

            # Add reflection insights
            if reflection:
                explanation_parts.append(
                    f"\nQuality Assessment: {reflection.quality_score:.2f}/1.0"
                )

                if reflection.issues:
                    explanation_parts.append("Issues Identified:")
                    for issue in reflection.issues:
                        explanation_parts.append(f"  - {issue}")

                if reflection.suggestions:
                    explanation_parts.append("Suggestions:")
                    for suggestion in reflection.suggestions:
                        explanation_parts.append(f"  - {suggestion}")

            # Add results summary
            recommendations = execution_result.get('recommendations', [])
            explanation_parts.append(
                f"\nResults: Found {len(recommendations)} recommendation(s)"
            )

            # Add the agent's natural explanation
            agent_explanation = execution_result.get('explanation', '')
            if agent_explanation:
                explanation_parts.append(f"\nAgent Response:\n{agent_explanation}")

            return "\n".join(explanation_parts)

        except Exception as e:
            logger.error(f"Explanation generation error: {e}")
            return f"Processed query: {query}\n\n{execution_result.get('explanation', '')}"

    def _update_memory(self,
                      query: str,
                      recommendations: List[Dict],
                      plan: Optional[ExecutionPlan]):
        """Update conversation memory with this turn."""
        try:
            # Extract recommendation titles
            titles = [rec.get('title', 'Unknown') for rec in recommendations]

            # Determine strategy used
            strategy = plan.strategy if plan else "direct"

            # Add to memory
            self.memory.add_turn(
                query=query,
                recommendations=titles,
                strategy=strategy
            )

        except Exception as e:
            logger.error(f"Memory update error: {e}")

    def _serialize_plan(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """Convert ExecutionPlan to serializable dictionary."""
        return {
            "query": plan.query,
            "strategy": plan.strategy,
            "estimated_quality": plan.estimated_quality,
            "steps": [
                {
                    "step_number": step.step_number,
                    "action": step.action,
                    "tool": step.tool,
                    "parameters": step.parameters,
                    "reasoning": step.reasoning,
                    "expected_output": step.expected_output
                }
                for step in plan.steps
            ]
        }

    def _serialize_reflection(self, reflection: Reflection) -> Dict[str, Any]:
        """Convert Reflection to serializable dictionary."""
        return {
            "success": reflection.success,
            "quality_score": reflection.quality_score,
            "issues": reflection.issues,
            "suggestions": reflection.suggestions,
            "should_retry": reflection.should_retry,
            "adjusted_parameters": reflection.adjusted_parameters
        }

    def get_conversation_history(self) -> List[Dict]:
        """Get the conversation history from memory."""
        if self.enable_memory and self.memory:
            return self.memory.get_recent_context(n=10)
        return []

    def clear_memory(self):
        """Clear conversation memory."""
        if self.enable_memory and self.memory:
            self.memory.clear()
            logger.info("Conversation memory cleared")


def main():
    """Test the AgenticRAGV2 system."""
    import os
    from dotenv import load_dotenv
    from src.utils.vector_store import initialize_vector_store

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("Error: OPENAI_API_KEY not found")
        return

    # Initialize vector store
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    print("Initializing vector store...")
    vector_store = initialize_vector_store(api_key, reset=False)

    # Initialize AgenticRAGV2
    print("\nInitializing AgenticRAGV2 system...")
    agentic_rag_v2 = AgenticRAGV2(
        vector_store=vector_store,
        openai_api_key=api_key,
        enable_planning=True,
        enable_reflection=True,
        enable_memory=True,
        verbose=True
    )

    # Test queries
    test_queries = [
        "I want a happy family movie under 2 hours",
        "Find me a scary horror movie for tonight",
        "Something like The Matrix - sci-fi action"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}/{len(test_queries)}: {query}")
        print('='*80)

        result = agentic_rag_v2.recommend(query)

        # Display results
        print(f"\nMethod: {result['method']}")
        print(f"Tool Calls: {result['tool_calls']}")

        # Planning trace
        if result.get('planning_trace'):
            plan = result['planning_trace']
            print(f"\nPlanning Strategy: {plan.get('strategy', 'N/A')}")
            print(f"Estimated Quality: {plan.get('estimated_quality', 0):.2f}")

            steps = plan.get('steps', [])
            if steps:
                print("\nPlanned Steps:")
                for step in steps:
                    print(f"  {step['step_number']}. {step['tool']}")
                    print(f"     Reasoning: {step['reasoning']}")

        # Execution trace
        if result.get('execution_trace'):
            print("\nExecution Trace:")
            for trace in result['execution_trace']:
                print(f"  {trace['step']}. {trace['tool']}")

        # Reflection
        if result.get('reflection_results'):
            reflection = result['reflection_results']
            print(f"\nReflection:")
            print(f"  Quality Score: {reflection.get('quality_score', 0):.2f}")
            print(f"  Success: {reflection.get('success', False)}")

            if reflection.get('issues'):
                print(f"  Issues: {', '.join(reflection['issues'])}")

            if reflection.get('suggestions'):
                print(f"  Suggestions: {', '.join(reflection['suggestions'])}")

        # Recommendations
        if result['recommendations']:
            print(f"\nRecommendations ({len(result['recommendations'])}):")
            for j, rec in enumerate(result['recommendations'], 1):
                title = rec.get('title', 'Unknown')
                year = rec.get('year', 'N/A')
                rating = rec.get('rating', 'N/A')
                print(f"{j}. {title} ({year}) - Rating: {rating}")

        # Detailed explanation
        print("\n" + "="*80)
        print("DETAILED EXPLANATION:")
        print("="*80)
        print(result['detailed_explanation'])

        # Conversation history
        history = agentic_rag_v2.get_conversation_history()
        print(f"\nConversation History: {len(history)} turn(s)")

    print("\n" + "="*80)
    print("Testing Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
