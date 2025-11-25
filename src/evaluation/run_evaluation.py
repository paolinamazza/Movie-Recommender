"""
Test runner for evaluating all three RAG systems.

Runs all 20 test queries through Naive, Advanced, and Agentic RAG systems,
collects results, calculates metrics, and saves to JSON files.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from dotenv import load_dotenv

from src.utils.vector_store import initialize_vector_store
from src.rag_systems.naive_rag import NaiveRAG
from src.rag_systems.advanced_rag import AdvancedRAG
from src.rag_systems.agentic_rag import AgenticRAG
from src.evaluation.metrics import EvaluationMetrics

logger = logging.getLogger(__name__)


class EvaluationRunner:
    """Runs evaluation across all RAG systems."""

    def __init__(self, openai_api_key: str, data_dir: Path, output_dir: Path):
        """
        Initialize the evaluation runner.

        Args:
            openai_api_key: OpenAI API key
            data_dir: Path to data directory
            output_dir: Path to output directory for results
        """
        self.openai_api_key = openai_api_key
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load test queries
        self.test_queries = self._load_test_queries()

        # Initialize vector store
        logger.info("Initializing vector store...")
        self.vector_store = initialize_vector_store(openai_api_key, reset=False)

        # Initialize RAG systems
        logger.info("Initializing RAG systems...")
        self.naive_rag = NaiveRAG(
            vector_store=self.vector_store,
            openai_api_key=openai_api_key,
            top_k=10
        )

        self.advanced_rag = AdvancedRAG(
            vector_store=self.vector_store,
            openai_api_key=openai_api_key,
            top_k=20,
            final_k=5
        )

        self.agentic_rag = AgenticRAG(
            vector_store=self.vector_store,
            openai_api_key=openai_api_key,
            verbose=False  # Reduce noise during evaluation
        )

        # Initialize metrics calculator
        content_dataset_path = self.data_dir / "processed" / "content_dataset.json"
        self.metrics_calculator = EvaluationMetrics(str(content_dataset_path))

        logger.info("Evaluation runner initialized")

    def _load_test_queries(self) -> List[Dict[str, Any]]:
        """Load all test queries (synthetic + manual)."""
        queries = []

        # Load synthetic queries
        synthetic_path = self.data_dir / "evaluation" / "test_queries.json"
        if synthetic_path.exists():
            with open(synthetic_path, 'r') as f:
                queries.extend(json.load(f))

        # Load manual queries
        manual_path = self.data_dir / "evaluation" / "manual_test_queries.json"
        if manual_path.exists():
            with open(manual_path, 'r') as f:
                queries.extend(json.load(f))

        logger.info(f"Loaded {len(queries)} test queries")
        return queries

    def run_single_query(self, rag_system, query_text: str, timeout: int = 60) -> Dict[str, Any]:
        """
        Run a single query through a RAG system with timeout.

        Args:
            rag_system: RAG system instance
            query_text: Query string
            timeout: Timeout in seconds

        Returns:
            Result dictionary
        """
        try:
            start_time = time.time()
            result = rag_system.recommend(query_text)
            elapsed = time.time() - start_time

            result['elapsed_time'] = elapsed
            result['success'] = True

            return result

        except Exception as e:
            logger.error(f"Error running query: {e}")
            return {
                'recommendations': [],
                'explanation': f"Error: {str(e)}",
                'method': 'error',
                'success': False,
                'elapsed_time': 0
            }

    def evaluate_system(self, system_name: str, rag_system) -> List[Dict[str, Any]]:
        """
        Evaluate a single RAG system on all test queries.

        Args:
            system_name: Name of the system (naive, advanced, agentic)
            rag_system: RAG system instance

        Returns:
            List of result dictionaries
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Evaluating {system_name.upper()} RAG")
        logger.info(f"{'='*80}\n")

        results = []

        for i, query_info in enumerate(self.test_queries, 1):
            query_text = query_info['query']
            query_id = query_info['id']

            logger.info(f"[{i}/{len(self.test_queries)}] Query {query_id}: {query_text[:60]}...")

            # Run query
            result = self.run_single_query(rag_system, query_text)

            # Calculate metrics
            metrics = self.metrics_calculator.calculate_all_metrics(
                result.get('recommendations', []),
                query_info
            )

            # Combine results
            full_result = {
                'query_id': query_id,
                'query': query_text,
                'difficulty': query_info.get('difficulty'),
                'expected_constraints': query_info.get('expected_constraints'),
                'recommendations': result.get('recommendations', []),
                'explanation': result.get('explanation', ''),
                'method': result.get('method', ''),
                'elapsed_time': result.get('elapsed_time', 0),
                'success': result.get('success', False),
                'metrics': metrics
            }

            # Add system-specific info
            if system_name == 'advanced':
                full_result['constraints_extracted'] = result.get('constraints', {})
            elif system_name == 'agentic':
                full_result['tool_trace'] = result.get('tool_trace', [])
                full_result['tool_calls'] = result.get('tool_calls', 0)

            results.append(full_result)

            # Log metrics
            logger.info(f"  Metrics: CSR={metrics['csr']:.3f}, P@5={metrics['precision_at_5']:.3f}, "
                       f"Mood={metrics.get('mood_match_accuracy', 0):.3f}, Time={result.get('elapsed_time', 0):.2f}s")

        return results

    def run_all_evaluations(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run evaluation on all three RAG systems.

        Returns:
            Dictionary mapping system names to results
        """
        all_results = {}

        # Evaluate Naive RAG
        all_results['naive'] = self.evaluate_system('naive', self.naive_rag)

        # Evaluate Advanced RAG
        all_results['advanced'] = self.evaluate_system('advanced', self.advanced_rag)

        # Evaluate Agentic RAG
        all_results['agentic'] = self.evaluate_system('agentic', self.agentic_rag)

        return all_results

    def save_results(self, all_results: Dict[str, List[Dict[str, Any]]]) -> None:
        """Save results to JSON files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for system_name, results in all_results.items():
            output_path = self.output_dir / f"results_{system_name}.json"

            # Save results
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved {system_name} results to {output_path}")

    def generate_summary(self, all_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Generate summary statistics across all systems.

        Args:
            all_results: Dictionary of system results

        Returns:
            Summary dictionary
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_queries': len(self.test_queries),
            'systems': {}
        }

        for system_name, results in all_results.items():
            # Aggregate metrics
            aggregated = self.metrics_calculator.aggregate_metrics(results)

            # Calculate success rate
            successful = sum(1 for r in results if r.get('success', False))
            success_rate = successful / len(results) if results else 0

            # Calculate average time
            avg_time = sum(r.get('elapsed_time', 0) for r in results) / len(results) if results else 0

            summary['systems'][system_name] = {
                'success_rate': success_rate,
                'average_time': avg_time,
                'metrics': aggregated
            }

        return summary

    def print_summary(self, summary: Dict[str, Any]) -> None:
        """Print formatted summary to console."""
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        print(f"\nTotal Queries: {summary['total_queries']}")
        print(f"Timestamp: {summary['timestamp']}\n")

        for system_name, stats in summary['systems'].items():
            print(f"\n{system_name.upper()} RAG:")
            print("-" * 40)
            print(f"Success Rate: {stats['success_rate']:.1%}")
            print(f"Average Time: {stats['average_time']:.2f}s")
            print(f"\nMetrics:")

            for metric_name, metric_stats in stats['metrics'].items():
                print(f"  {metric_name}:")
                print(f"    Mean: {metric_stats['mean']:.3f}")
                print(f"    Std:  {metric_stats['std']:.3f}")
                print(f"    Min:  {metric_stats['min']:.3f}")
                print(f"    Max:  {metric_stats['max']:.3f}")

        print("\n" + "="*80)


def main():
    """Main execution function."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load environment
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment")
        return

    # Setup paths
    base_dir = Path(__file__).parent.parent.parent
    data_dir = base_dir / "data"
    output_dir = data_dir / "evaluation"

    # Run evaluation
    runner = EvaluationRunner(api_key, data_dir, output_dir)

    logger.info("\nStarting evaluation...")
    logger.info("This will take approximately 10-20 minutes depending on API latency.\n")

    all_results = runner.run_all_evaluations()

    # Save results
    runner.save_results(all_results)

    # Generate and save summary
    summary = runner.generate_summary(all_results)
    summary_path = output_dir / "metrics_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nSaved summary to {summary_path}")

    # Print summary
    runner.print_summary(summary)


if __name__ == "__main__":
    main()
