"""
🎬 CineMatch AI - Beautiful Cinema-Themed Streamlit Interface

A stunning movie theater-inspired interface for comparing RAG architectures.
Features premium design with gold/red cinema colors, smooth animations, and interactive visualizations.
"""

import streamlit as st
import sys
from pathlib import Path
import os
import time

# --- OPTIONAL IMPORTS (Graceful Degradation) ---
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    go = None
    PLOTLY_AVAILABLE = False

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# --- PROJECT SETUP ---
# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag_systems.naive_rag import NaiveRAG
from src.rag_systems.advanced_rag import AdvancedRAG
from src.rag_systems.agentic_rag import AgenticRAG
from src.utils.vector_store import initialize_vector_store

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="CineMatch AI",
    page_icon="🍿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS (The "Beautiful" Part) ---
st.markdown("""
<style>
    /* 1. FONTS & RESET */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=Bebas+Neue&display=swap');

    :root {
        --bg-color: #0a0a0a;
        --card-bg: #161616;
        --accent-red: #E50914;
        --accent-gold: #E5C100;
        --text-primary: #FFFFFF;
        --text-secondary: #A3A3A3;
        --glass: rgba(255, 255, 255, 0.05);
    }

    /* Remove Streamlit default padding/margins to make it look like a website */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 5rem;
        max-width: 95% !important;
    }

    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* 2. HERO SECTION */
    .hero-container {
        text-align: center;
        padding: 4rem 0 2rem 0;
        background: radial-gradient(circle at center, rgba(229, 9, 20, 0.15) 0%, rgba(10, 10, 10, 0) 70%);
        margin-bottom: 2rem;
    }

    .main-title {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 5rem;
        line-height: 1;
        background: linear-gradient(180deg, #FFFFFF 0%, #A3A3A3 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: 2px;
    }

    .subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        color: var(--accent-red);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 4px;
    }

    /* 3. SYSTEM COLUMNS & HEADERS */
    .system-header {
        background: var(--card-bg);
        border-top: 3px solid;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }

    .sys-naive { border-color: #CD7F32; }
    .sys-advanced { border-color: #C0C0C0; }
    .sys-agentic { border-color: #E5C100; }

    .sys-name {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 1.8rem;
        color: var(--text-primary);
        margin: 0;
    }

    .sys-badge {
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        text-transform: uppercase;
        padding: 4px 8px;
        border-radius: 4px;
        background: rgba(255,255,255,0.1);
        color: var(--text-secondary);
    }

    /* 4. MOVIE CARDS */
    .movie-card {
        background: var(--card-bg);
        border-radius: 12px;
        padding: 0;
        margin-bottom: 1.2rem;
        overflow: hidden;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid rgba(255,255,255,0.05);
        position: relative;
    }

    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.5);
        border-color: rgba(255,255,255,0.2);
    }

    .movie-header {
        padding: 1rem;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
    }

    .movie-title {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 1.1rem;
        color: white;
        line-height: 1.3;
    }

    .movie-year {
        color: var(--text-secondary);
        font-size: 0.9rem;
        font-weight: 400;
    }

    .rating-badge {
        background: rgba(229, 193, 0, 0.15);
        color: var(--accent-gold);
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 800;
        font-size: 0.85rem;
    }

    .movie-body {
        padding: 1rem;
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        color: #CCCCCC;
    }

    .genre-tag {
        display: inline-block;
        font-size: 0.7rem;
        padding: 3px 8px;
        margin-right: 5px;
        margin-bottom: 5px;
        border-radius: 100px;
        background: rgba(255,255,255,0.1);
        color: var(--text-secondary);
    }

    /* 5. TRACE TERMINAL (For Agentic) */
    .trace-terminal {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 1rem;
        font-family: 'Courier New', monospace;
        font-size: 0.8rem;
        color: #58a6ff;
        margin-top: 1rem;
        max-height: 200px;
        overflow-y: auto;
    }

    .trace-step {
        margin-bottom: 0.8rem;
        border-bottom: 1px solid #21262d;
        padding-bottom: 0.5rem;
    }

    .trace-tool { color: #ff7b72; font-weight: bold; }
    .trace-action { color: #d2a8ff; }
    .trace-result { color: #8b949e; font-style: italic; }

    /* 6. UI ELEMENTS */
    .stTextInput > div > div > input {
        background-color: #1A1A1A;
        color: white;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 1rem;
        font-size: 1.1rem;
    }

    .stTextInput > div > div > input:focus {
        border-color: var(--accent-red);
        box-shadow: 0 0 0 1px var(--accent-red);
    }

    .stButton > button {
        background-color: var(--accent-red) !important;
        color: white !important;
        border: none !important;
        font-weight: 700 !important;
        padding: 0.8rem 2rem !important;
        font-size: 1rem !important;
        transition: all 0.2s;
        border-radius: 8px !important;
        width: 100%;
    }

    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 20px rgba(229, 9, 20, 0.4);
    }

    /* Metrics */
    .metric-box {
        background: rgba(255,255,255,0.03);
        border-radius: 6px;
        padding: 0.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-val { font-weight: bold; color: white; }
    .metric-lbl { font-size: 0.7rem; color: var(--text-secondary); }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0F0F0F;
    }

    /* Explanation box */
    .explanation-box {
        background: rgba(255,255,255,0.05);
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
        border-left: 3px solid #666;
    }

    .explanation-label {
        color: #aaa;
        text-transform: uppercase;
        font-size: 0.7rem;
        margin-bottom: 0.5rem;
    }

    .explanation-text {
        color: #ddd;
        font-size: 0.9rem;
        line-height: 1.5;
    }

</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

@st.cache_resource(show_spinner=False)
def load_rag_systems():
    """Initialize all three RAG systems (cached)."""
    if DOTENV_AVAILABLE:
        load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("🔑 OPENAI_API_KEY not found. Please set it in .env or environment variables.")
        st.stop()

    vector_store = initialize_vector_store(api_key, reset=False)

    return (
        NaiveRAG(vector_store=vector_store, openai_api_key=api_key, top_k=10),
        AdvancedRAG(vector_store=vector_store, openai_api_key=api_key, top_k=20, final_k=5),
        AgenticRAG(vector_store=vector_store, openai_api_key=api_key, verbose=False)
    )


def render_movie_card(item, rank):
    """Generates HTML for a single beautiful movie card."""
    title = item.get('title', 'Unknown')
    year = item.get('year', 'N/A')
    rating = item.get('tmdb_rating', 0)
    genres = item.get('genres', [])[:3]
    moods = item.get('moods', [])[:3]
    runtime = item.get('total_hours', 0)
    item_type = item.get('type', 'Movie').title()

    # Generate tags HTML
    tags_html = "".join([f'<span class="genre-tag">{g}</span>' for g in genres])
    moods_text = ', '.join(moods) if moods else 'No moods tagged'

    return f"""
    <div class="movie-card">
        <div class="movie-header">
            <div>
                <div class="movie-title">{rank}. {title} <span class="movie-year">({year})</span></div>
            </div>
            <div class="rating-badge">★ {rating:.1f}</div>
        </div>
        <div class="movie-body">
            <div style="margin-bottom: 8px; color: #888; font-size: 0.8rem;">
                ⏱ {runtime:.1f}h &nbsp;•&nbsp; {item_type}
            </div>
            <div style="margin-bottom: 8px;">{tags_html}</div>
            <div style="font-size: 0.85rem; color: #aaa; font-style: italic;">
                "{moods_text}"
            </div>
        </div>
    </div>
    """


def render_tool_trace(trace):
    """Generates HTML for the 'Matrix/Terminal' style trace."""
    if not trace:
        return ""

    steps_html = ""
    for step in trace:
        tool = step.get('tool', 'System')
        tool_input = step.get('input', 'N/A')
        output = step.get('output_summary', 'N/A')

        steps_html += f"""
        <div class="trace-step">
            <span class="trace-tool"> > {tool}</span><br>
            <span class="trace-action">   Input: {tool_input}</span><br>
            <span class="trace-result">   Output: {output}</span>
        </div>
        """

    return f'<div class="trace-terminal">{steps_html}<div style="color:#58a6ff;">_</div></div>'


# --- MAIN APP LOGIC ---

def main():
    # 1. Load Systems
    with st.spinner('🍿 Warming up the popcorn machine (Loading Models)...'):
        try:
            naive_rag, advanced_rag, agentic_rag = load_rag_systems()
        except Exception as e:
            st.error(f"Error loading systems: {e}")
            st.stop()

    # 2. Sidebar (Settings only)
    with st.sidebar:
        st.markdown("### ⚙️ CONFIGURATION")

        st.markdown("**RAG Systems:**")
        run_naive = st.checkbox("🥉 Naive RAG", value=True, help="Fast & Direct - Bronze Tier")
        run_advanced = st.checkbox("🥈 Advanced RAG", value=True, help="Smart Reranking - Silver Tier")
        run_agentic = st.checkbox("🥇 Agentic RAG", value=True, help="Multi-Step Reasoning - Gold Tier")

        st.markdown("---")
        st.markdown("**Display Options:**")
        show_trace = st.toggle("Show Agent Thinking", value=True)
        show_metrics = st.toggle("Show Latency Stats", value=True)
        n_recs = st.slider("Results Count", 1, 10, 5)

        st.markdown("---")
        st.markdown("### 📊 System Comparison")

        with st.expander("🥉 Bronze - Naive RAG"):
            st.markdown("""
            **Fast & Simple**
            - ⚡ 2-3s response time
            - 💰 ~$0.005/query
            - 🎯 Direct vector search
            - Best for: Quick lookups
            """)

        with st.expander("🥈 Silver - Advanced RAG"):
            st.markdown("""
            **Balanced Performance**
            - ⚖️ 4-5s response time
            - 💰 ~$0.012/query
            - 🎯 HyDE + Reranking
            - Best for: Multi-criteria
            """)

        with st.expander("🥇 Gold - Agentic RAG"):
            st.markdown("""
            **Premium Intelligence**
            - 🎬 8-12s response time
            - 💰 ~$0.025/query
            - 🎯 7 AI-powered tools
            - Best for: Complex queries
            """)

        st.markdown("---")
        st.info("**CineMatch AI v2.0**\n\n1,964 Movies & Series\nOpenAI Embeddings\nGPT-4o Intelligence")

    # 3. Hero Area
    st.markdown("""
        <div class="hero-container">
            <div class="subtitle">Intelligent Recommendation Engine</div>
            <div class="main-title">CINEMATCH AI</div>
        </div>
    """, unsafe_allow_html=True)

    # 4. Search Interface (Combined Pills + Input)

    # Suggestion Pills
    st.markdown("<div style='text-align:center; margin-bottom: 10px; color: #666;'>Quick Select Mood:</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    # Session state for query
    if 'query_input' not in st.session_state:
        st.session_state.query_input = ""

    if c1.button("😂 Funny Comedy", use_container_width=True):
        st.session_state.query_input = "I want a funny comedy movie"
    if c2.button("😱 Scary Horror", use_container_width=True):
        st.session_state.query_input = "Scary horror movie for tonight"
    if c3.button("🤯 Mind-Bending", use_container_width=True):
        st.session_state.query_input = "Mind-bending thriller like Inception"
    if c4.button("👨‍👩‍👧 Family Night", use_container_width=True):
        st.session_state.query_input = "Family-friendly movie under 2 hours"

    # The Main Search Bar
    query = st.text_input(
        "Search",
        value=st.session_state.query_input,
        placeholder="Describe your perfect movie night... (e.g. 'Romantic comedy with a twist')",
        label_visibility="collapsed"
    )

    # Update session state when user types
    query_stripped = query.strip()
    if query_stripped:
        st.session_state.query_input = query

    # Center the Search Button
    _, col_btn, _ = st.columns([1, 2, 1])
    run_search = col_btn.button("✨ FIND MOVIES", type="primary")

    # 5. Results Processing
    if run_search:
        if not query_stripped:
            st.warning("⚠️ Please describe what you feel like watching so we can search for it.")
            st.stop()

        st.markdown("---")

        st.markdown("---")

        # Prepare the columns based on selection
        active_systems = []
        if run_naive:
            active_systems.append(('naive', naive_rag, "Naive RAG", "sys-naive", "🥉 Bronze: Fast & Direct"))
        if run_advanced:
            active_systems.append(('advanced', advanced_rag, "Advanced RAG", "sys-advanced", "🥈 Silver: Smart Reranking"))
        if run_agentic:
            active_systems.append(('agentic', agentic_rag, "Agentic RAG", "sys-agentic", "🥇 Gold: AI-Powered"))

        if not active_systems:
            st.warning("⚠️ Please enable at least one system in the sidebar.")
            st.stop()

        cols = st.columns(len(active_systems))
        results_data = {}  # For charts

        # Iterate through systems
        for idx, (sys_key, sys_obj, name, css_class, badge_text) in enumerate(active_systems):
            with cols[idx]:
                # System Header
                st.markdown(f"""
                <div class="system-header {css_class}">
                    <div class="sys-name">{name}</div>
                    <span class="sys-badge">{badge_text}</span>
                </div>
                """, unsafe_allow_html=True)

                # Logic Execution
                with st.spinner("Searching..."):
                    start_t = time.time()
                    try:
                        response = sys_obj.recommend(query_stripped)
                        elapsed = time.time() - start_t

                        results_data[name] = elapsed

                        # Metrics Display
                        if show_metrics:
                            mc1, mc2 = st.columns(2)
                            cost = 0.005 if sys_key == 'naive' else 0.012 if sys_key == 'advanced' else 0.025
                            mc1.markdown(f"""<div class="metric-box"><div class="metric-val">{elapsed:.2f}s</div><div class="metric-lbl">LATENCY</div></div>""", unsafe_allow_html=True)
                            mc2.markdown(f"""<div class="metric-box"><div class="metric-val">${cost:.3f}</div><div class="metric-lbl">EST. COST</div></div>""", unsafe_allow_html=True)

                        # Trace (Agent Only)
                        if sys_key == 'agentic' and show_trace:
                            tool_trace = response.get('tool_trace', [])
                            if tool_trace:
                                with st.expander(f"🔎 View Agent Thinking Process ({len(tool_trace)} steps)", expanded=False):
                                    trace_html = render_tool_trace(tool_trace)
                                    st.markdown(trace_html, unsafe_allow_html=True)

                        # Recommendations List
                        recs = response.get('recommendations', [])[:n_recs]

                        if not recs:
                            st.info("🎬 No movies found. Try adjusting your query!")

                        for rank, item in enumerate(recs, 1):
                            st.markdown(render_movie_card(item, rank), unsafe_allow_html=True)

                        # Explanation (if any)
                        explanation = response.get('explanation', '')
                        if explanation and len(explanation) > 50:
                            with st.expander("💬 AI Reasoning", expanded=False):
                                st.markdown(f"""
                                <div class="explanation-box">
                                    <div class="explanation-label">Why these recommendations?</div>
                                    <div class="explanation-text">{explanation}</div>
                                </div>
                                """, unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"❌ System Error: {e}")
                        results_data[name] = 0

        # 6. Comparison Chart (Bottom)
        if len(results_data) > 1 and PLOTLY_AVAILABLE and show_metrics:
            st.markdown("---")
            st.markdown("<h3 style='text-align:center; font-family:Bebas Neue; color:#888; margin-top:3rem;'>⚡ PERFORMANCE BENCHMARK</h3>", unsafe_allow_html=True)

            names = list(results_data.keys())
            times = list(results_data.values())
            colors = ['#CD7F32', '#C0C0C0', '#FFD700'][:len(names)]

            fig = go.Figure(data=[go.Bar(
                x=names,
                y=times,
                marker_color=colors,
                text=[f"{t:.2f}s" for t in times],
                textposition='auto'
            )])

            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#AAAAAA', family='Inter'),
                margin=dict(l=20, r=20, t=20, b=20),
                height=300,
                yaxis_title="Response Time (seconds)",
                xaxis_title="RAG System"
            )

            st.plotly_chart(fig, use_container_width=True)

            # Detailed comparison table
            st.markdown("### 📊 Detailed Comparison")
            comparison_data = []
            for name, elapsed in results_data.items():
                tier = "Bronze" if "Naive" in name else "Silver" if "Advanced" in name else "Gold"
                icon = "🥉" if "Naive" in name else "🥈" if "Advanced" in name else "🥇"
                cost = 0.005 if "Naive" in name else 0.012 if "Advanced" in name else 0.025

                comparison_data.append({
                    'System': f"{icon} {name}",
                    'Tier': tier,
                    'Latency': f"{elapsed:.2f}s",
                    'Est. Cost': f"${cost:.3f}"
                })

            if comparison_data:
                st.table(comparison_data)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; color: #666;">
        <div style="font-size: 0.9rem; margin-bottom: 0.5rem;">
            Powered by OpenAI GPT-4o • ChromaDB • LangChain
        </div>
        <div style="font-size: 0.8rem;">
            🥉 Bronze: Naive RAG | 🥈 Silver: Advanced RAG | 🥇 Gold: Agentic RAG
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
