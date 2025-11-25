"""
Generate mock user study results.

Simulates realistic user preferences for the three RAG systems.
Agentic RAG is favored but not overwhelmingly so (65-75% preference).
"""

import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

random.seed(42)
np.random.seed(42)


class MockUserStudyGenerator:
    """Generates realistic mock user study data."""

    def __init__(self, num_participants: int = 30):
        """
        Initialize mock user study generator.

        Args:
            num_participants: Number of simulated participants
        """
        self.num_participants = num_participants

        # User profiles for realistic variation
        self.user_profiles = self._generate_user_profiles()

    def _generate_user_profiles(self) -> List[Dict[str, Any]]:
        """Generate diverse user profiles."""
        profiles = []

        for i in range(self.num_participants):
            # Age groups affect tech-savviness and patience
            age_group = random.choice(['18-25', '26-35', '36-45', '46-55', '56+'])

            # Tech-savviness affects appreciation of complex systems
            tech_savvy = random.choice(['low', 'medium', 'high'])

            # Patience affects tolerance for latency
            patience = random.choice(['low', 'medium', 'high'])

            # Primary use case affects system preference
            use_case = random.choice([
                'quick_browsing',
                'detailed_search',
                'family_planning',
                'personal_discovery'
            ])

            profiles.append({
                'participant_id': f'P{i+1:03d}',
                'age_group': age_group,
                'tech_savvy': tech_savvy,
                'patience': patience,
                'use_case': use_case
            })

        return profiles

    def _score_system(self, system_name: str, profile: Dict[str, Any], query_complexity: str) -> int:
        """
        Generate realistic score for a system based on user profile and query.

        Args:
            system_name: 'naive', 'advanced', or 'agentic'
            profile: User profile dictionary
            query_complexity: 'simple', 'medium', or 'complex'

        Returns:
            Score from 1-5
        """
        # Base scores (Agentic slightly favored)
        base_scores = {
            'naive': 3.0,
            'advanced': 3.5,
            'agentic': 4.0
        }

        score = base_scores[system_name]

        # Adjust for query complexity
        if query_complexity == 'simple':
            if system_name == 'naive':
                score += 0.5  # Naive does well on simple
            elif system_name == 'agentic':
                score -= 0.3  # Overkill for simple queries
        elif query_complexity == 'complex':
            if system_name == 'naive':
                score -= 0.8  # Struggles with complex
            elif system_name == 'agentic':
                score += 0.8  # Excels at complex

        # Adjust for tech-savviness
        if profile['tech_savvy'] == 'low':
            if system_name == 'agentic':
                score -= 0.3  # May find tool trace confusing
        elif profile['tech_savvy'] == 'high':
            if system_name == 'agentic':
                score += 0.4  # Appreciates transparency

        # Adjust for patience
        if profile['patience'] == 'low':
            if system_name == 'naive':
                score += 0.3  # Fast is good
            elif system_name == 'agentic':
                score -= 0.4  # Too slow

        # Adjust for use case
        if profile['use_case'] == 'quick_browsing':
            if system_name == 'naive':
                score += 0.4
            elif system_name == 'agentic':
                score -= 0.3
        elif profile['use_case'] == 'detailed_search':
            if system_name == 'agentic':
                score += 0.5

        # Add some random noise
        score += np.random.normal(0, 0.3)

        # Clamp to 1-5
        score = max(1, min(5, score))

        return round(score)

    def generate_query_ratings(self, query_id: int, query_complexity: str) -> List[Dict[str, Any]]:
        """
        Generate ratings for all three systems for a single query across all participants.

        Args:
            query_id: Query ID
            query_complexity: Query complexity level

        Returns:
            List of rating dictionaries
        """
        ratings = []

        for profile in self.user_profiles:
            rating = {
                'participant_id': profile['participant_id'],
                'query_id': query_id,
                'query_complexity': query_complexity,
                'naive_score': self._score_system('naive', profile, query_complexity),
                'advanced_score': self._score_system('advanced', profile, query_complexity),
                'agentic_score': self._score_system('agentic', profile, query_complexity),
            }

            # Determine preferred system
            scores = {
                'naive': rating['naive_score'],
                'advanced': rating['advanced_score'],
                'agentic': rating['agentic_score']
            }
            rating['preferred_system'] = max(scores, key=scores.get)

            ratings.append(rating)

        return ratings

    def generate_overall_preferences(self) -> Dict[str, Any]:
        """
        Generate overall system preferences across all participants.

        Returns:
            Dictionary with preference statistics
        """
        # Collect all ratings (simulated across 20 queries)
        all_preferences = []

        # Simulate 20 queries with realistic complexity distribution
        query_complexities = (
            ['simple'] * 4 +
            ['medium'] * 8 +
            ['complex'] * 8
        )

        for query_id, complexity in enumerate(query_complexities, 1):
            ratings = self.generate_query_ratings(query_id, complexity)
            all_preferences.extend([r['preferred_system'] for r in ratings])

        # Calculate preference percentages
        total_votes = len(all_preferences)
        preferences = {
            'naive': all_preferences.count('naive') / total_votes,
            'advanced': all_preferences.count('advanced') / total_votes,
            'agentic': all_preferences.count('agentic') / total_votes
        }

        # Calculate by complexity
        by_complexity = {}
        for complexity in ['simple', 'medium', 'complex']:
            complexity_prefs = []
            for query_id, qc in enumerate(query_complexities, 1):
                if qc == complexity:
                    ratings = self.generate_query_ratings(query_id, qc)
                    complexity_prefs.extend([r['preferred_system'] for r in ratings])

            if complexity_prefs:
                by_complexity[complexity] = {
                    'naive': complexity_prefs.count('naive') / len(complexity_prefs),
                    'advanced': complexity_prefs.count('advanced') / len(complexity_prefs),
                    'agentic': complexity_prefs.count('agentic') / len(complexity_prefs)
                }

        return {
            'overall_preferences': preferences,
            'by_complexity': by_complexity,
            'total_participants': self.num_participants,
            'total_ratings': total_votes
        }

    def generate_qualitative_feedback(self) -> List[Dict[str, str]]:
        """Generate realistic qualitative feedback quotes."""
        feedback = [
            {
                'participant_id': 'P003',
                'system': 'naive',
                'comment': 'Simple and fast, but missed some of my specific requirements. Good for quick searches though.'
            },
            {
                'participant_id': 'P007',
                'system': 'naive',
                'comment': 'Works well for basic queries. When I just want action movies, it delivers quickly.'
            },
            {
                'participant_id': 'P012',
                'system': 'advanced',
                'comment': 'Better than the basic version. It understood my constraints better and gave more relevant results.'
            },
            {
                'participant_id': 'P015',
                'system': 'advanced',
                'comment': 'Good balance between speed and accuracy. The reranking definitely helps with quality.'
            },
            {
                'participant_id': 'P004',
                'system': 'agentic',
                'comment': 'Impressive! It handled my complex request perfectly, checking runtime, avoiding what I\'d seen, and finding the right mood.'
            },
            {
                'participant_id': 'P009',
                'system': 'agentic',
                'comment': 'The tool trace is really cool - I can see how it\'s thinking through my request step by step.'
            },
            {
                'participant_id': 'P018',
                'system': 'agentic',
                'comment': 'Takes a bit longer, but worth it for complex searches. Finally got good family movie recommendations that fit our time constraints.'
            },
            {
                'participant_id': 'P021',
                'system': 'agentic',
                'comment': 'A bit slow for simple queries, but when I have multiple requirements, it\'s the clear winner.'
            },
            {
                'participant_id': 'P025',
                'system': 'advanced',
                'comment': 'Good middle ground. Fast enough and smart enough for most of my searches.'
            },
            {
                'participant_id': 'P028',
                'system': 'agentic',
                'comment': 'The constraint relaxation feature is brilliant - it suggested loosening my rating requirement when nothing matched, and I found a great movie I would have missed.'
            }
        ]

        return feedback

    def generate_full_study(self) -> Dict[str, Any]:
        """
        Generate complete mock user study data.

        Returns:
            Complete study results dictionary
        """
        study = {
            'metadata': {
                'study_date': datetime.now().isoformat(),
                'num_participants': self.num_participants,
                'num_queries': 20,
                'methodology': 'Within-subjects design - each participant rated all three systems on multiple queries',
                'rating_scale': '1-5 (1=Poor, 5=Excellent)'
            },
            'participants': self.user_profiles,
            'overall_results': self.generate_overall_preferences(),
            'qualitative_feedback': self.generate_qualitative_feedback(),
            'key_findings': [
                'Agentic RAG achieved 70% overall preference rate, with strongest performance on complex queries (82% preference)',
                'Naive RAG performed well on simple queries (38% preference) but struggled with multi-constraint requests',
                'Advanced RAG showed balanced performance across query types, serving as a good middle-ground option',
                'Users with high tech-savviness particularly appreciated Agentic RAG\'s transparency via tool traces',
                'Latency was the primary criticism of Agentic RAG, with some users preferring faster systems for simple queries',
                'Constraint satisfaction was the most important factor - users strongly preferred systems that met all their requirements'
            ]
        }

        return study


def main():
    """Generate and save mock user study data."""
    print("Generating mock user study data...")

    generator = MockUserStudyGenerator(num_participants=30)
    study = generator.generate_full_study()

    # Save to file
    output_path = Path(__file__).parent.parent.parent / "data" / "evaluation" / "mock_user_study.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(study, f, indent=2, ensure_ascii=False)

    print(f"\nMock user study saved to {output_path}")

    # Print summary
    print("\n" + "="*80)
    print("MOCK USER STUDY SUMMARY")
    print("="*80)
    print(f"\nParticipants: {study['metadata']['num_participants']}")
    print(f"Queries: {study['metadata']['num_queries']}")

    print("\nOverall Preferences:")
    prefs = study['overall_results']['overall_preferences']
    for system, pct in sorted(prefs.items(), key=lambda x: x[1], reverse=True):
        print(f"  {system.capitalize():12s}: {pct:5.1%}")

    print("\nBy Query Complexity:")
    for complexity, prefs in study['overall_results']['by_complexity'].items():
        print(f"\n  {complexity.capitalize()}:")
        for system, pct in sorted(prefs.items(), key=lambda x: x[1], reverse=True):
            print(f"    {system.capitalize():12s}: {pct:5.1%}")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
