"""
Evaluation metrics for RAG systems.

Implements:
- Constraint Satisfaction Rate (CSR)
- Precision@5
- Mood Match Accuracy
- Runtime Fit Quality
"""

import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """Calculates evaluation metrics for recommendation systems."""

    def __init__(self, content_dataset_path: Optional[str] = None):
        """
        Initialize evaluation metrics calculator.

        Args:
            content_dataset_path: Path to content dataset for ground truth
        """
        self.content_dataset = None
        if content_dataset_path:
            self.load_content_dataset(content_dataset_path)

    def load_content_dataset(self, path: str) -> None:
        """Load content dataset for ground truth lookups."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.content_dataset = {str(item['id']): item for item in data}
        logger.info(f"Loaded {len(self.content_dataset)} items for evaluation")

    def calculate_csr(self,
                     recommendations: List[Dict[str, Any]],
                     constraints: Dict[str, Any]) -> float:
        """
        Calculate Constraint Satisfaction Rate.

        CSR = (Total constraints satisfied) / (Total constraints * num_recommendations)

        Args:
            recommendations: List of recommended items
            constraints: Dictionary of constraints to check

        Returns:
            CSR score (0-1)
        """
        if not recommendations or not constraints:
            return 0.0

        total_satisfied = 0
        total_possible = len(constraints) * len(recommendations)

        for item in recommendations:
            satisfied = self._check_constraints(item, constraints)
            total_satisfied += sum(satisfied.values())

        csr = total_satisfied / total_possible if total_possible > 0 else 0.0
        logger.debug(f"CSR: {csr:.3f} ({total_satisfied}/{total_possible})")
        return csr

    def calculate_precision_at_k(self,
                                 recommendations: List[Dict[str, Any]],
                                 constraints: Dict[str, Any],
                                 k: int = 5) -> float:
        """
        Calculate Precision@K.

        P@K = (Number of acceptable recommendations in top K) / K

        An item is "acceptable" if it satisfies at least 70% of constraints.

        Args:
            recommendations: List of recommended items
            constraints: Dictionary of constraints
            k: Number of top results to consider

        Returns:
            Precision@K score (0-1)
        """
        if not recommendations or not constraints:
            return 0.0

        top_k = recommendations[:k]
        acceptable_count = 0

        for item in top_k:
            satisfied = self._check_constraints(item, constraints)
            satisfaction_rate = sum(satisfied.values()) / len(satisfied) if satisfied else 0
            if satisfaction_rate >= 0.7:
                acceptable_count += 1

        precision = acceptable_count / k if k > 0 else 0.0
        logger.debug(f"P@{k}: {precision:.3f} ({acceptable_count}/{k} acceptable)")
        return precision

    def calculate_mood_match_accuracy(self,
                                     recommendations: List[Dict[str, Any]],
                                     target_mood: str) -> float:
        """
        Calculate Mood Match Accuracy.

        MMA = (Number of items with target mood) / (Total items)

        Args:
            recommendations: List of recommended items
            target_mood: Target mood string

        Returns:
            Mood match accuracy (0-1)
        """
        if not recommendations or not target_mood:
            return 0.0

        target_mood = target_mood.lower()
        matches = 0

        for item in recommendations:
            item_moods = [m.lower() for m in item.get('moods', [])]
            if target_mood in item_moods:
                matches += 1

        accuracy = matches / len(recommendations) if recommendations else 0.0
        logger.debug(f"Mood Match Accuracy: {accuracy:.3f} ({matches}/{len(recommendations)})")
        return accuracy

    def calculate_runtime_fit_quality(self,
                                      recommendations: List[Dict[str, Any]],
                                      max_hours: Optional[float] = None,
                                      target_range: Optional[tuple] = None) -> float:
        """
        Calculate Runtime Fit Quality.

        RFQ measures how well recommendations fit time constraints.

        Scoring:
        - If max_hours specified: penalty for items exceeding limit
        - If target_range specified: reward for items in range
        - Score = average fit score across all items

        Args:
            recommendations: List of recommended items
            max_hours: Maximum hours constraint
            target_range: Tuple of (min_hours, max_hours) for ideal range

        Returns:
            Runtime fit quality score (0-1)
        """
        if not recommendations:
            return 0.0

        scores = []

        for item in recommendations:
            total_hours = item.get('total_hours', 0)
            runtime_minutes = item.get('runtime_minutes', 0)

            # For series, use episode runtime
            if item.get('type') == 'series' and runtime_minutes > 0:
                episode_hours = runtime_minutes / 60.0
            else:
                episode_hours = total_hours

            score = 1.0

            # Check max_hours constraint
            if max_hours is not None:
                if item.get('type') == 'movie':
                    if total_hours > max_hours:
                        # Penalty based on how much over
                        excess_ratio = (total_hours - max_hours) / max_hours
                        score *= max(0, 1.0 - excess_ratio)
                    elif total_hours >= max_hours * 0.7:
                        # Perfect fit
                        score *= 1.0
                    else:
                        # Too short, minor penalty
                        score *= 0.9
                else:  # series
                    if episode_hours > max_hours:
                        score *= 0.5  # Episode too long
                    else:
                        score *= 1.0  # Episode fits

            # Check target_range
            if target_range is not None:
                min_h, max_h = target_range
                if item.get('type') == 'series':
                    # Check episode runtime
                    episode_hours = runtime_minutes / 60.0 if runtime_minutes else 0.75
                    if min_h <= episode_hours <= max_h:
                        score *= 1.0
                    else:
                        score *= 0.5
                else:
                    # Check movie runtime
                    if min_h <= total_hours <= max_h:
                        score *= 1.0
                    elif total_hours < min_h:
                        score *= 0.7
                    else:
                        score *= 0.5

            scores.append(score)

        rfq = np.mean(scores) if scores else 0.0
        logger.debug(f"Runtime Fit Quality: {rfq:.3f}")
        return rfq

    def calculate_all_metrics(self,
                             recommendations: List[Dict[str, Any]],
                             query_info: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate all metrics for a set of recommendations.

        Args:
            recommendations: List of recommended items
            query_info: Dictionary with query, constraints, expected_constraints, etc.

        Returns:
            Dictionary of metric scores
        """
        constraints = query_info.get('expected_constraints', {})

        metrics = {
            'csr': 0.0,
            'precision_at_5': 0.0,
            'mood_match_accuracy': 0.0,
            'runtime_fit_quality': 0.0
        }

        if not recommendations:
            logger.warning("No recommendations to evaluate")
            return metrics

        # CSR
        metrics['csr'] = self.calculate_csr(recommendations, constraints)

        # Precision@5
        metrics['precision_at_5'] = self.calculate_precision_at_k(recommendations, constraints, k=5)

        # Mood Match Accuracy
        if 'mood' in constraints:
            metrics['mood_match_accuracy'] = self.calculate_mood_match_accuracy(
                recommendations,
                constraints['mood']
            )

        # Runtime Fit Quality
        max_hours = constraints.get('max_hours')
        runtime_range = None
        if 'runtime_minutes_range' in constraints:
            r = constraints['runtime_minutes_range']
            runtime_range = (r[0]/60.0, r[1]/60.0)

        if max_hours or runtime_range:
            metrics['runtime_fit_quality'] = self.calculate_runtime_fit_quality(
                recommendations,
                max_hours=max_hours,
                target_range=runtime_range
            )

        return metrics

    def _check_constraints(self, item: Dict[str, Any], constraints: Dict[str, Any]) -> Dict[str, bool]:
        """
        Check which constraints an item satisfies.

        Args:
            item: Content item
            constraints: Constraints to check

        Returns:
            Dictionary mapping constraint names to satisfaction status
        """
        satisfied = {}

        # Mood constraint
        if 'mood' in constraints:
            target_mood = constraints['mood'].lower()
            item_moods = [m.lower() for m in item.get('moods', [])]
            satisfied['mood'] = target_mood in item_moods

        # Type constraint
        if 'type' in constraints:
            satisfied['type'] = item.get('type') == constraints['type']

        # Genre constraint
        if 'genre' in constraints:
            target_genres = constraints['genre']
            if isinstance(target_genres, str):
                target_genres = [target_genres]
            target_genres = [g.lower() for g in target_genres]
            item_genres = [g.lower() for g in item.get('genres', [])]
            satisfied['genre'] = any(tg in item_genres for tg in target_genres)

        # Rating constraint
        if 'min_rating' in constraints:
            item_rating = item.get('tmdb_rating', 0)
            satisfied['min_rating'] = item_rating >= constraints['min_rating']

        # Runtime constraint
        if 'max_hours' in constraints:
            total_hours = item.get('total_hours', 0)
            if item.get('type') == 'movie':
                satisfied['max_hours'] = total_hours <= constraints['max_hours']
            else:
                # For series, check episode runtime
                runtime_min = item.get('runtime_minutes', 45)
                episode_hours = runtime_min / 60.0
                satisfied['max_hours'] = episode_hours <= constraints['max_hours']

        # Context constraint
        if 'context' in constraints:
            context_suit = item.get('context_suitability', {})
            if isinstance(context_suit, str):
                context_suit = json.loads(context_suit)
            score = context_suit.get(constraints['context'], 0)
            satisfied['context'] = score >= 0.6

        # Content rating constraint
        if 'content_rating' in constraints:
            allowed = constraints['content_rating']
            if isinstance(allowed, list):
                satisfied['content_rating'] = item.get('content_rating') in allowed
            else:
                satisfied['content_rating'] = item.get('content_rating') == allowed

        return satisfied

    def aggregate_metrics(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate metrics across multiple queries.

        Args:
            all_results: List of result dictionaries with 'metrics' field

        Returns:
            Dictionary with mean, std, min, max for each metric
        """
        metrics_by_name = {}

        # Collect all metric values
        for result in all_results:
            metrics = result.get('metrics', {})
            for name, value in metrics.items():
                if name not in metrics_by_name:
                    metrics_by_name[name] = []
                if value is not None:
                    metrics_by_name[name].append(value)

        # Calculate statistics
        aggregated = {}
        for name, values in metrics_by_name.items():
            if values:
                aggregated[name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'count': len(values)
                }
            else:
                aggregated[name] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'count': 0
                }

        return aggregated


def main():
    """Test the metrics module."""
    # Sample data
    recommendations = [
        {
            'id': 1,
            'type': 'movie',
            'title': 'Test Movie 1',
            'moods': ['happy', 'excited'],
            'genres': ['Comedy', 'Action'],
            'tmdb_rating': 8.0,
            'total_hours': 1.8,
            'context_suitability': {'family_watch': 0.9, 'solo_watch': 0.8}
        },
        {
            'id': 2,
            'type': 'movie',
            'title': 'Test Movie 2',
            'moods': ['happy'],
            'genres': ['Comedy'],
            'tmdb_rating': 7.5,
            'total_hours': 1.5,
            'context_suitability': {'family_watch': 0.85, 'solo_watch': 0.7}
        }
    ]

    constraints = {
        'mood': 'happy',
        'type': 'movie',
        'min_rating': 7.0,
        'max_hours': 2.0,
        'context': 'family_watch'
    }

    query_info = {
        'query': 'I want a happy family movie under 2 hours',
        'expected_constraints': constraints
    }

    # Calculate metrics
    logging.basicConfig(level=logging.INFO)
    evaluator = EvaluationMetrics()
    metrics = evaluator.calculate_all_metrics(recommendations, query_info)

    print("\nEvaluation Metrics:")
    print(f"CSR: {metrics['csr']:.3f}")
    print(f"Precision@5: {metrics['precision_at_5']:.3f}")
    print(f"Mood Match Accuracy: {metrics['mood_match_accuracy']:.3f}")
    print(f"Runtime Fit Quality: {metrics['runtime_fit_quality']:.3f}")


if __name__ == "__main__":
    main()
