# CineMatch AI: RAG Architecture Comparison for Movie Recommendations

A comparative analysis of three Retrieval-Augmented Generation (RAG) architectures applied to movie and TV series recommendations. This system demonstrates the evolution from simple vector search to autonomous agent-based retrieval.

## Authors

- Paolina Beatrice Mazza
- Chiara Canali

## Overview

This project implements and evaluates three distinct RAG paradigms for content recommendation:

1. **Baseline (Naive RAG)**: Direct vector similarity search using semantic embeddings
2. **Advanced RAG**: Enhanced with HyDE query expansion and cross-encoder reranking
3. **Agentic RAG**: Autonomous agent with specialized tools using the ReAct framework

The system demonstrates how architectural complexity impacts recommendation quality, particularly for multi-constraint queries.

## Key Features

- Mood-based content recommendation (10 emotional categories)
- Context-aware filtering (solo watch, family watch, date night, etc.)
- Multi-constraint query handling
- Real-time comparative analysis across three RAG architectures
- Interactive web interface with live demonstrations

## Dataset

- **Source**: TMDB (The Movie Database) API v3
- **Size**: 2,000 items (1,000 movies + 1,000 series)
- **Enrichments**:
  - Mood classification using genre and keyword mapping
  - Context suitability scores for different viewing scenarios
  - Runtime and rating metadata
  - Cast and crew information

## System Requirements

- Python 3.9 or higher
- OpenAI API key
- 2GB free disk space

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd movie_recommender_final
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=your-api-key-here
```

## Quick Start

### Initialize Vector Store

Before first use, initialize the vector database with embeddings:

```bash
PYTHONPATH=. python3 -c "
from src.utils.vector_store import initialize_vector_store
import os
from dotenv import load_dotenv

load_dotenv()
vs = initialize_vector_store(os.getenv('OPENAI_API_KEY'), reset=True)
print(f'Vector store initialized with {vs.count()} items')
"
```

**Note**: This step costs approximately $0.02 in API fees and takes 1-2 minutes.

### Launch Web Interface

Start the web application:

```bash
chmod +x run_web.sh
./run_web.sh
```

The application will open automatically at `http://localhost:5001`

## Architecture

### Baseline RAG (Naive)
- Direct cosine similarity search on embedded content
- Single-turn LLM generation
- Optimized for speed (~2.3s per query)
- Cost: ~$0.005 per query

### Advanced RAG
- HyDE (Hypothetical Document Embeddings) for query expansion
- Cross-encoder reranking for precision improvement
- Two-stage retrieval pipeline
- Latency: ~4.2s per query
- Cost: ~$0.012 per query

### Agentic RAG
- ReAct framework for autonomous decision-making
- Seven specialized tools:
  - `mood_content_finder`: Semantic mood-based search
  - `similar_content_finder`: Find similar titles
  - `runtime_matcher`: Hard runtime constraints
  - `viewing_history_checker`: Exclude watched content
  - `context_filter`: Context suitability scoring
  - `general_search_tool`: Fallback search
  - `planner`: Multi-step query decomposition
- Dynamic tool orchestration
- Latency: ~9.5s per query
- Cost: ~$0.032 per query

## Project Structure

```
movie_recommender_final/
├── app/
│   ├── templates/
│   │   └── index.html          # Web interface
│   └── web_app.py              # Flask application
├── src/
│   ├── preprocessing/
│   │   └── data_processor.py   # Dataset preprocessing
│   ├── rag_systems/
│   │   ├── naive_rag.py        # Baseline implementation
│   │   ├── advanced_rag.py     # HyDE + Reranking
│   │   └── agentic_rag.py      # ReAct agent
│   ├── tools/                  # Agent tools
│   ├── planning/               # Query planner
│   └── utils/
│       ├── vector_store.py     # ChromaDB interface
│       └── config.py           # Configuration
├── data/
│   ├── raw/                    # Original TMDB data
│   ├── processed/              # Enriched dataset
│   └── chroma_db/             # Vector embeddings
├── requirements.txt
├── .env.example
└── run_web.sh
```

## Usage Examples

### Simple Mood Query
```
"Find me a happy movie for tonight"
```
**Result**: All three systems perform similarly, Baseline is fastest.

### Multi-Constraint Query
```
"Recommend a thriller under 2 hours with rating above 7.5, 
but NOT The Twilight Zone because I already watched it"
```
**Result**: Agentic RAG excels by enforcing hard constraints via tool chaining.

### Context-Specific Query
```
"Find something perfect for background watch while I work, 
rated above 8.0"
```
**Result**: Advanced and Agentic RAG better understand nuanced context.

## Evaluation Metrics

- **Constraint Satisfaction Rate (CSR)**: Percentage of user-defined constraints met
- **Latency**: End-to-end response time
- **Cost**: API usage per query
- **Qualitative Assessment**: Relevance and diversity of recommendations

## Results Summary

| Metric | Baseline | Advanced | Agentic |
|--------|----------|----------|---------|
| Simple Queries CSR | 85% | 88% | 90% |
| Complex Queries CSR | 42% | 63% | 88% |
| Avg Latency | 2.3s | 4.2s | 9.5s |
| Cost per Query | $0.005 | $0.012 | $0.032 |

**Key Finding**: Architectural complexity is justified for multi-constraint queries where Agentic RAG achieves 88% CSR compared to 42% for Baseline.

## Technologies

- **LLM**: OpenAI GPT-4
- **Embeddings**: text-embedding-3-small (1536 dimensions)
- **Vector Database**: ChromaDB
- **Framework**: LangChain (for Agentic RAG)
- **Web Framework**: Flask
- **Frontend**: HTML/CSS/JavaScript

## Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=your-api-key-here

# Optional (defaults shown)
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o
TEMPERATURE=0.7
CHROMA_PERSIST_DIR=./data/chroma_db
LOG_LEVEL=INFO
```

## Limitations

- Content ratings coverage: 41.3% (US-specific ratings from TMDB)
- Dataset limited to popular English-language content (68%)
- Agentic RAG latency may exceed 10s for complex queries
- API costs scale with query complexity

## Future Work

- Integration of user feedback for personalization
- Collaborative filtering for improved similarity
- Support for real-time TMDB updates
- Optimization of Agentic RAG tool selection
- Multi-modal recommendations (posters, trailers)

## License

This project is for academic and research purposes.

## Acknowledgments

Dataset sourced from The Movie Database (TMDB) API.
