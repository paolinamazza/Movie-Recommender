"""
Planner: Creates multi-step execution plans for complex queries.
"""

import logging
from typing import Dict, List, Any
from dataclasses import dataclass
from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class PlanStep:
    """Represents a single step in the execution plan."""
    step_number: int
    action: str
    tool: str
    parameters: Dict[str, Any]
    reasoning: str
    expected_output: str


@dataclass
class ExecutionPlan:
    """Complete execution plan for a query."""
    query: str
    strategy: str
    steps: List[PlanStep]
    estimated_quality: float


class Planner:
    """Creates strategic execution plans for movie recommendation queries."""

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.available_tools = [
            "mood_content_finder",
            "context_filter",
            "runtime_matcher",
            "viewing_history_checker",
            "similar_content_finder",
            "general_search_tool",
            "constraint_relaxer",
            "group_preference_reconciler"
        ]

    def create_plan(self, query: str, conversation_context: List[Dict] = None) -> ExecutionPlan:
        """
        Create a strategic execution plan for the query.

        Args:
            query: User's movie recommendation query
            conversation_context: Previous conversation history (optional)

        Returns:
            ExecutionPlan with ordered steps
        """
        logger.info(f"Creating execution plan for: {query}")

        # Analyze query and create plan
        planning_prompt = self._build_planning_prompt(query, conversation_context)

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at planning movie recommendation strategies."},
                    {"role": "user", "content": planning_prompt}
                ],
                temperature=0.3
            )

            plan_text = response.choices[0].message.content
            plan = self._parse_plan(query, plan_text)

            logger.info(f"Created plan with {len(plan.steps)} steps using '{plan.strategy}' strategy")
            return plan

        except Exception as e:
            logger.error(f"Planning error: {e}")
            # Fallback to simple plan
            return self._create_fallback_plan(query)

    def _build_planning_prompt(self, query: str, conversation_context: List[Dict] = None) -> str:
        """Build the planning prompt."""
        prompt = f"""Analyze this movie recommendation query and create an optimal execution plan.

Query: "{query}"

Available Tools:
1. mood_content_finder - Search by emotional mood (happy, sad, excited, etc.)
2. context_filter - Filter by viewing context (family_watch, date_night, solo_watch, etc.)
3. runtime_matcher - Match to available time constraints
4. viewing_history_checker - Avoid previously watched content (REQUIRES items from another tool first)
6. general_search_tool - Semantic search for topics/genres (e.g. "space movies", "90s action")
7. constraint_relaxer - Suggest relaxations when no results found
8. group_preference_reconciler - Reconcile multiple users' preferences

Query Analysis Guidelines:
- If query mentions mood/emotion → use mood_content_finder first
- If query mentions specific title/movie → use similar_content_finder
- If query mentions "like X" → use similar_content_finder
- If query is general (e.g. "space movies", "action films") → use general_search_tool
- If query mentions time constraint → include runtime_matcher
- If query mentions viewing history → use viewing_history_checker (but get items first!)
- If query mentions multiple people/group → use group_preference_reconciler
- viewing_history_checker MUST come AFTER getting items from another tool

Create a plan with 2-4 steps maximum. Format your response EXACTLY like this:

STRATEGY: [one of: mood_based, similarity_based, constrained_search, group_recommendation]

STEPS:
1. TOOL: tool_name
   REASONING: Why this tool and why first
   PARAMETERS: {{"param": "value"}}
   EXPECTED: What we expect to get

2. TOOL: tool_name
   REASONING: Why this tool second
   PARAMETERS: {{"param": "value"}}
   EXPECTED: What we expect to get

ESTIMATED_QUALITY: 0.85 (0.0-1.0 confidence in this plan)
"""

        if conversation_context:
            prompt += f"\n\nConversation Context:\n{conversation_context}\n"

        return prompt

    def _parse_plan(self, query: str, plan_text: str) -> ExecutionPlan:
        """Parse the LLM's plan response."""
        lines = plan_text.strip().split('\n')

        strategy = "mood_based"  # default
        steps = []
        estimated_quality = 0.8

        current_step = None

        for line in lines:
            line = line.strip()

            if line.startswith("STRATEGY:"):
                strategy = line.split(":", 1)[1].strip()

            elif line.startswith("ESTIMATED_QUALITY:"):
                try:
                    estimated_quality = float(line.split(":", 1)[1].strip().split()[0])
                except:
                    estimated_quality = 0.8

            elif line and line[0].isdigit() and "." in line[:3]:
                # Start of a new step
                if current_step:
                    steps.append(current_step)
                current_step = {
                    "step_number": len(steps) + 1,
                    "action": "",
                    "tool": "",
                    "parameters": {},
                    "reasoning": "",
                    "expected_output": ""
                }

            elif current_step:
                if line.startswith("TOOL:"):
                    tool_name = line.split(":", 1)[1].strip()
                    # Clean up tool name if it has extra text
                    if tool_name:
                        current_step["tool"] = tool_name
                        current_step["action"] = f"Use {tool_name}"
                elif line.startswith("REASONING:"):
                    current_step["reasoning"] = line.split(":", 1)[1].strip()
                elif line.startswith("PARAMETERS:"):
                    # Simple parameter parsing
                    param_text = line.split(":", 1)[1].strip()
                    try:
                        import json
                        current_step["parameters"] = json.loads(param_text)
                    except:
                        current_step["parameters"] = {}
                elif line.startswith("EXPECTED:"):
                    current_step["expected_output"] = line.split(":", 1)[1].strip()

        if current_step:
            steps.append(current_step)

        # Convert to PlanStep objects
        plan_steps = []
        for step_data in steps:
            tool_name = step_data.get("tool", "")
            # If tool is still empty, try to infer from reasoning
            if not tool_name and step_data.get("reasoning"):
                reasoning_lower = step_data["reasoning"].lower()
                for available_tool in self.available_tools:
                    if available_tool.replace("_", " ") in reasoning_lower or available_tool in reasoning_lower:
                        tool_name = available_tool
                        break

            plan_steps.append(PlanStep(
                step_number=step_data["step_number"],
                action=f"Use {tool_name}" if tool_name else "Execute step",
                tool=tool_name,
                parameters=step_data.get("parameters", {}),
                reasoning=step_data.get("reasoning", ""),
                expected_output=step_data.get("expected_output", "")
            ))

        return ExecutionPlan(
            query=query,
            strategy=strategy,
            steps=plan_steps,
            estimated_quality=estimated_quality
        )

    def _create_fallback_plan(self, query: str) -> ExecutionPlan:
        """Create a simple fallback plan if planning fails."""
        logger.warning("Using fallback plan")

        steps = [
            PlanStep(
                step_number=1,
                action="Search for content",
                tool="general_search_tool",
                parameters={"query": query},
                reasoning="Fallback: General search for content",
                expected_output="List of movies"
            )
        ]

        return ExecutionPlan(
            query=query,
            strategy="fallback",
            steps=steps,
            estimated_quality=0.5
        )
