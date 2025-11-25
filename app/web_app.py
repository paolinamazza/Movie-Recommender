"""
CineMatch AI - Flask Web Application

A professional HTML-based alternative to the Streamlit interface.
Provides cinematic, interactive movie recommendations with RAG comparison.
"""

import os
import sys
import json
import logging
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.vector_store import initialize_vector_store
from src.rag_systems.naive_rag import NaiveRAG
from src.rag_systems.advanced_rag import AdvancedRAG
from src.rag_systems.agentic_rag_v2 import AgenticRAGV2

# Load environment
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__,
            static_folder='static',
            template_folder='templates')
CORS(app)
app.config['JSON_SORT_KEYS'] = False

# Global RAG systems (initialized on first request)
vector_store = None
naive_rag = None
advanced_rag = None
agentic_rag = None
initialized = False


def initialize_systems():
    """Initialize all RAG systems (lazy loading)."""
    global vector_store, naive_rag, advanced_rag, agentic_rag, initialized

    if initialized:
        return

    logger.info("🎬 Initializing CineMatch AI systems...")

    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment")

    # Initialize vector store
    logger.info("Loading vector store...")
    vector_store = initialize_vector_store(api_key, reset=False)
    logger.info(f"✅ Vector store loaded: {vector_store.count()} items")

    # Initialize RAG systems
    logger.info("Initializing Naive RAG...")
    naive_rag = NaiveRAG(vector_store, api_key)

    logger.info("Initializing Advanced RAG...")
    advanced_rag = AdvancedRAG(vector_store, api_key)

    logger.info("Initializing Agentic RAG (with planning & reflection)...")
    agentic_rag = AgenticRAGV2(vector_store, api_key)

    initialized = True
    logger.info("✅ All systems initialized!")


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/api/search', methods=['POST'])
def search():
    """
    Handle search requests from frontend.

    Request JSON:
    {
        "query": "happy family movie",
        "systems": ["naive", "advanced", "agentic"],
        "showMetrics": true
    }

    Response JSON:
    {
        "success": true,
        "results": {
            "naive": {...},
            "advanced": {...},
            "agentic": {...}
        }
    }
    """
    try:
        # Initialize systems if needed
        initialize_systems()

        # Parse request
        data = request.get_json()
        query = data.get('query', '').strip()
        systems = data.get('systems', ['naive', 'advanced', 'agentic'])
        show_metrics = data.get('showMetrics', True)

        if not query:
            return jsonify({
                'success': False,
                'error': 'Query cannot be empty'
            }), 400

        logger.info(f"🔍 Search query: '{query}' | Systems: {systems}")

        # Execute searches with timing
        import time
        results = {}

        if 'naive' in systems:
            logger.info("Running Naive RAG...")
            start_time = time.time()
            naive_result = naive_rag.recommend(query)
            latency = time.time() - start_time
            results['naive'] = format_result(naive_result, 'Naive RAG', show_metrics, latency)

        if 'advanced' in systems:
            logger.info("Running Advanced RAG...")
            start_time = time.time()
            advanced_result = advanced_rag.recommend(query)
            latency = time.time() - start_time
            results['advanced'] = format_result(advanced_result, 'Advanced RAG', show_metrics, latency)

        if 'agentic' in systems:
            logger.info("Running Agentic RAG (with planning & reflection)...")
            start_time = time.time()
            agentic_result = agentic_rag.recommend(query)
            latency = time.time() - start_time
            logger.info(f"Agentic RAG returned {len(agentic_result.get('recommendations', []))} recommendations")
            if len(agentic_result.get('recommendations', [])) > 0:
                logger.info(f"First recommendation: {agentic_result['recommendations'][0].get('title', 'Unknown')}")
            results['agentic'] = format_result(agentic_result, 'Agentic RAG', show_metrics, latency)

        return jsonify({
            'success': True,
            'query': query,
            'results': results
        })

    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/search/<system>', methods=['POST'])
def search_single_system(system):
    """
    Handle search for a single RAG system (for progressive loading).

    Request JSON:
    {
        "query": "happy family movie"
    }
    """
    try:
        # Initialize systems if needed
        initialize_systems()

        # Parse request
        data = request.get_json()
        query = data.get('query', '').strip()

        if not query:
            return jsonify({
                'success': False,
                'error': 'Query cannot be empty'
            }), 400

        if system not in ['naive', 'advanced', 'agentic']:
            return jsonify({
                'success': False,
                'error': f'Invalid system: {system}'
            }), 400

        logger.info(f"🔍 Single system query: '{query}' | System: {system}")

        # Execute search with timing
        import time
        start_time = time.time()

        if system == 'naive':
            result = naive_rag.recommend(query)
            system_name = 'Naive RAG'
        elif system == 'advanced':
            result = advanced_rag.recommend(query)
            system_name = 'Advanced RAG'
        else:  # agentic
            result = agentic_rag.recommend(query)
            system_name = 'Agentic RAG'

        latency = time.time() - start_time
        formatted_result = format_result(result, system_name, True, latency)

        return jsonify({
            'success': True,
            'query': query,
            'system': system,
            'result': formatted_result
        })

    except Exception as e:
        logger.error(f"Single system search error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/stats', methods=['GET'])
def stats():
    """Get system statistics."""
    try:
        initialize_systems()

        return jsonify({
            'success': True,
            'stats': {
                'total_items': vector_store.count() if vector_store else 0,
                'systems_ready': initialized,
                'available_systems': ['naive', 'advanced', 'agentic']
            }
        })

    except Exception as e:
        logger.error(f"Stats error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def format_result(result, system_name, show_metrics=True, latency=0):
    """Format RAG result for JSON response."""
    formatted = {
        'system': system_name,
        'recommendations': result.get('recommendations', []),
        'explanation': result.get('explanation', ''),
        'method': result.get('method', ''),
        'count': len(result.get('recommendations', []))
    }

    if show_metrics:
        # Estimate tokens and cost (rough approximations)
        # GPT-4o: $2.50/1M input, $10/1M output
        # Embedding: $0.02/1M tokens

        # Estimate based on system complexity
        if 'Naive' in system_name:
            est_input_tokens = 500
            est_output_tokens = 200
        elif 'Advanced' in system_name:
            est_input_tokens = 800  # HyDE + reranking
            est_output_tokens = 300
        else:  # Agentic (with planning, reflection, memory)
            tool_calls = result.get('tool_calls', 0)
            # Agentic has planning + execution + reflection
            est_input_tokens = 1500 + (tool_calls * 400)
            est_output_tokens = 600 + (tool_calls * 200)

        # Calculate cost
        input_cost = (est_input_tokens / 1_000_000) * 2.50
        output_cost = (est_output_tokens / 1_000_000) * 10.00
        total_cost = input_cost + output_cost

        formatted['metrics'] = {
            'latency': round(latency, 2),
            'latency_ms': round(latency * 1000),
            'tool_calls': result.get('tool_calls', 0),
            'reranked': result.get('reranked', False),
            'hyde_used': result.get('hyde_used', False),
            'est_tokens': est_input_tokens + est_output_tokens,
            'est_cost': round(total_cost, 4)
        }

    # Add tool trace for agentic
    if 'tool_trace' in result:
        formatted['tool_trace'] = result['tool_trace']

    # Add V2-specific traces
    if 'planning_trace' in result:
        formatted['planning_trace'] = result['planning_trace']
    if 'execution_trace' in result:
        formatted['execution_trace'] = result['execution_trace']
    if 'reflection_results' in result:
        formatted['reflection_results'] = result['reflection_results']
    if 'detailed_explanation' in result:
        formatted['detailed_explanation'] = result['detailed_explanation']

    return formatted


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'initialized': initialized
    })


if __name__ == '__main__':
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║                                                            ║
    ║              🎬 CineMatch AI Web Interface 🎬              ║
    ║                                                            ║
    ║         Professional HTML Alternative to Streamlit        ║
    ║                                                            ║
    ╚════════════════════════════════════════════════════════════╝

    🚀 Starting Flask server...
    🌐 Open browser to: http://localhost:5001

    Press CTRL+C to stop
    """)

    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True,
        use_reloader=False  # Avoid double initialization
    )
