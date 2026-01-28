"""
Ultimate Trie Search Optimization - Streamlit Application
Complete comparison of 4 algorithms with beautiful UI

Author: DSA Project
Features: 
- Google-style search dropdown
- 4 algorithm comparison (Naive, Standard Trie, Compressed Trie, TST)
- Real-time visualization
- Comprehensive benchmarking
- Analytics dashboard
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time
import json
import os
from typing import List, Dict

# Import all implementations
from trie import Trie, SearchResult
from compressed_trie import CompressedTrie
from ternary_trie import TernarySearchTrie

# Import benchmark - FIX: Use ComprehensiveBenchmark instead of PerformanceBenchmark
try:
    from benchmark import ComprehensiveBenchmark, NaiveSearch
except ImportError:
    # Fallback if import fails
    class NaiveSearch:
        def __init__(self):
            self.words = []
        
        def insert(self, word, frequency=1):
            word = word.lower().strip()
            for item in self.words:
                if item['word'] == word:
                    item['frequency'] = frequency
                    return
            self.words.append({'word': word, 'frequency': frequency})
        
        def search(self, prefix, limit=10):
            prefix = prefix.lower().strip()
            matches = [item for item in self.words if item['word'].startswith(prefix)]
            matches.sort(key=lambda x: x['frequency'], reverse=True)
            return matches[:limit]

# Page configuration
st.set_page_config(
    page_title="Trie Optimization Project - 4 Algorithms",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        font-size: 18px;
        font-weight: 600;
    }
    .algo-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .algo-card-naive {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    .algo-card-standard {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    .algo-card-compressed {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
    }
    .algo-card-tst {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    }
    .suggestion-item {
        padding: 12px;
        margin: 8px 0;
        border-radius: 8px;
        font-size: 16px;
        font-weight: 500;
    }
    .green-suggestion {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .yellow-suggestion {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    .orange-suggestion {
        background-color: #f8d7da;
        border-left: 4px solid #fd7e14;
    }
    .dropdown-container {
        background: white;
        border: 1px solid #dfe1e5;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-top: -10px;
        padding: 8px 0;
    }
    .suggestion-row {
        padding: 12px 16px;
        cursor: pointer;
        transition: background 0.2s;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .suggestion-row:hover {
        background-color: #f8f9fa;
    }
    </style>
    """, unsafe_allow_html=True)


# Initialize session state
if 'standard_trie' not in st.session_state:
    st.session_state.standard_trie = Trie(max_suggestions=10)
    st.session_state.compressed_trie = CompressedTrie(max_suggestions=10)
    st.session_state.tst = TernarySearchTrie(max_suggestions=10)
    st.session_state.naive = NaiveSearch()
    st.session_state.search_history = []
    st.session_state.data_loaded = False
    st.session_state.dataset_size = None
    st.session_state.benchmark_results = None

def load_words_from_file(filepath: str, max_words: int = None):
    """
    Load words from a text file with realistic frequency distribution.
    
    Args:
        filepath: Path to the word file
        max_words: Maximum number of words to load (None = all)
    
    Returns:
        List of (word, frequency) tuples
    """
    words = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            for i, line in enumerate(lines):
                if max_words and i >= max_words:
                    break
                
                word = line.strip().lower()
                if word and word.isalpha():  # Only alphabetic words
                    # Zipf distribution for realistic frequencies
                    # Most popular words have higher frequencies
                    frequency = max(1, int(10000 / ((i + 1) ** 0.7)))
                    words.append((word, frequency))
        
        return words
    
    except FileNotFoundError:
        st.error(f"❌ File not found: {filepath}")
        return None
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        return None


def load_dataset(dataset_size: str):
    """
    Load dataset of specified size into all 4 algorithms.
    
    Args:
        dataset_size: '1k', '10k', or '100k'
    """
    # Map size to filepath
    file_map = {
        '1k': 'data/words_1k.txt',
        '10k': 'data/words_10k.txt',
        '100k': 'data/words_100k.txt'
    }
    
    filepath = file_map.get(dataset_size)
    if not filepath:
        st.error(f"Invalid dataset size: {dataset_size}")
        return False
    
    # Load words from file
    words = load_words_from_file(filepath)
    
    if words:
        with st.spinner(f"Loading {len(words):,} words into all 4 algorithms..."):
            # Clear existing data
            st.session_state.standard_trie = Trie(max_suggestions=10)
            st.session_state.compressed_trie = CompressedTrie(max_suggestions=10)
            st.session_state.tst = TernarySearchTrie(max_suggestions=10)
            st.session_state.naive = NaiveSearch()
            
            # Insert all words
            for word, freq in words:
                st.session_state.standard_trie.insert(word, freq)
                st.session_state.compressed_trie.insert(word, freq)
                st.session_state.tst.insert(word, freq)
                st.session_state.naive.insert(word, freq)
            
            st.session_state.data_loaded = True
            st.session_state.dataset_size = dataset_size
            return True
    
    return False


@st.cache_data
def load_benchmark_results():
    """Load benchmark results from JSON file"""
    try:
        with open('results/benchmark_results.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def display_algo_card(title: str, time_ms: float, color_class: str, extra_info: str = ""):
    """Display an algorithm performance card"""
    st.markdown(f"""
        <div class="algo-card {color_class}">
            <h3>{title}</h3>
            <h2>{time_ms:.3f} ms</h2>
            <p>{extra_info}</p>
        </div>
    """, unsafe_allow_html=True)


def create_comparison_chart(naive_time, trie_time, comp_time, tst_time):
    """Create bar chart comparing all 4 algorithms"""
    fig = go.Figure()
    
    algorithms = ['Naive', 'Standard Trie', 'Compressed Trie', 'TST']
    times = [naive_time, trie_time, comp_time, tst_time]
    colors = ['#f5576c', '#00f2fe', '#38f9d7', '#fee140']
    
    fig.add_trace(go.Bar(
        x=algorithms,
        y=times,
        marker_color=colors,
        text=[f"{t:.3f}ms" for t in times],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Algorithm Performance Comparison",
        yaxis_title="Response Time (ms)",
        height=400,
        template='plotly_white'
    )
    
    return fig


def create_benchmark_performance_chart(benchmark_data):
    """Create performance chart from benchmark results"""
    summary = benchmark_data.get('summary', {})
    
    sizes = []
    naive_times = []
    trie_times = []
    comp_times = []
    tst_times = []
    
    for size_str in sorted(summary.keys(), key=lambda x: int(x)):
        size = int(size_str)
        data = summary[size_str]
        
        sizes.append(size)
        naive_times.append(data.get('Naive', {}).get('avg_time_ms', 0))
        trie_times.append(data.get('Standard Trie', {}).get('avg_time_ms', 0))
        comp_times.append(data.get('Compressed Trie', {}).get('avg_time_ms', 0))
        tst_times.append(data.get('TST', {}).get('avg_time_ms', 0))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=sizes, y=naive_times,
        mode='lines+markers',
        name='Naive',
        line=dict(color='red', width=3),
        marker=dict(size=10)
    ))
    
    fig.add_trace(go.Scatter(
        x=sizes, y=trie_times,
        mode='lines+markers',
        name='Standard Trie',
        line=dict(color='blue', width=3),
        marker=dict(size=10)
    ))
    
    fig.add_trace(go.Scatter(
        x=sizes, y=comp_times,
        mode='lines+markers',
        name='Compressed Trie',
        line=dict(color='green', width=3),
        marker=dict(size=10)
    ))
    
    fig.add_trace(go.Scatter(
        x=sizes, y=tst_times,
        mode='lines+markers',
        name='TST',
        line=dict(color='purple', width=3),
        marker=dict(size=10)
    ))
    
    fig.update_layout(
        title="Response Time vs Dataset Size",
        xaxis_title="Dataset Size (words)",
        yaxis_title="Response Time (ms)",
        xaxis_type="log",
        height=500,
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def create_speedup_chart(benchmark_data):
    """Create speedup comparison chart"""
    summary = benchmark_data.get('summary', {})
    
    speedups = []
    for size_str in sorted(summary.keys(), key=lambda x: int(x)):
        data = summary[size_str]
        
        if 'Naive' in data:
            naive_time = data['Naive']['avg_time_ms']
            
            speedups.append({
                'Dataset Size': f"{int(size_str):,}",
                'Standard Trie': naive_time / data['Standard Trie']['avg_time_ms'] if 'Standard Trie' in data else 0,
                'Compressed Trie': naive_time / data['Compressed Trie']['avg_time_ms'] if 'Compressed Trie' in data else 0,
                'TST': naive_time / data['TST']['avg_time_ms'] if 'TST' in data else 0
            })
    
    df = pd.DataFrame(speedups)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['Dataset Size'],
        y=df['Standard Trie'],
        name='Standard Trie',
        marker_color='blue'
    ))
    
    fig.add_trace(go.Bar(
        x=df['Dataset Size'],
        y=df['Compressed Trie'],
        name='Compressed Trie',
        marker_color='green'
    ))
    
    fig.add_trace(go.Bar(
        x=df['Dataset Size'],
        y=df['TST'],
        name='TST',
        marker_color='purple'
    ))
    
    fig.update_layout(
        title="Speedup Factors vs Naive Search",
        xaxis_title="Dataset Size",
        yaxis_title="Speedup Factor (x)",
        barmode='group',
        height=400,
        template='plotly_white'
    )
    
    return fig


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.title("⚙️ Trie Optimization")
    st.markdown("### 4 Algorithm Comparison")
    
    # Dataset selector
    st.markdown("### 📚 Select Dataset")
    
    dataset_options = {
        '1k': '1,000 words (Fast)',
        '10k': '10,000 words (Medium)',
        '100k': '100,000 words (Large)'
    }
    
    selected_size = st.selectbox(
        "Choose dataset size:",
        options=list(dataset_options.keys()),
        format_func=lambda x: dataset_options[x],
        key='dataset_selector'
    )
    
    if st.button("📥 Load Dataset", use_container_width=True, type="primary"):
        success = load_dataset(selected_size)
        if success:
            word_count = st.session_state.standard_trie.get_stats()['total_words']
            st.success(f"✅ Loaded {word_count:,} words!")
            st.balloons()
    
    # Show current dataset info
    if st.session_state.data_loaded:
        st.info(f"📊 Currently loaded: **{dataset_options[st.session_state.dataset_size]}**")
    
    st.markdown("---")
    
    # Statistics comparison
    if st.session_state.data_loaded:
        st.subheader("📊 Statistics")
        
        standard_stats = st.session_state.standard_trie.get_stats()
        comp_stats = st.session_state.compressed_trie.get_stats()
        tst_stats = st.session_state.tst.get_stats()
        
        st.metric("Total Words", f"{standard_stats['total_words']:,}")
        
        st.markdown("**Nodes:**")
        st.text(f"Standard: {standard_stats['total_nodes']:,}")
        st.text(f"Compressed: {comp_stats['total_nodes']:,}")
        st.text(f"TST: {tst_stats['total_nodes']:,}")
        
        st.markdown("**Space Savings:**")
        st.text(f"Compressed: {comp_stats['space_saved']}")
        st.text(f"TST: {tst_stats['memory_savings']}")
    
    st.markdown("---")
    
    # About
    st.subheader("ℹ️ About")
    st.markdown("""
    **4 Algorithms Compared:**
    
    🐌 **Naive** - O(n×m)
    Baseline linear search
    
    ⚡ **Standard Trie** - O(p)
    Fast, memory intensive
    
    🎯 **Compressed Trie** - O(p)
    60-70% space reduction
    
    💾 **TST** - O(p+log n)
    89% memory savings
    """)

# ============================================================================
# MAIN APP
# ============================================================================

# Header
st.title("🔍 Trie Search Optimization Project")
st.markdown("### Comparing 4 Algorithms: Naive, Standard Trie, Compressed Trie, and TST")

# Show data loading prompt if no data loaded
if not st.session_state.data_loaded:
    st.info("👈 **Select a dataset size in the sidebar and click 'Load Dataset' to begin!**")
    st.markdown("""
    **Available datasets:**
    - 📦 **1,000 words** - Quick demo (loads in seconds)
    - 📦 **10,000 words** - Medium dataset (good for testing)
    - 📦 **100,000 words** - Large dataset (impressive for presentation!)
    """)

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Overview",
    "🔍 Live Search Comparison", 
    "🌳 Trie Visualization", 
    "📊 Benchmarks",
    "📈 Dashboard"
])

# ============================================================================
# TAB 1: OVERVIEW
# ============================================================================
with tab1:
    st.header("Project Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Objective
        Design and implement an efficient real-time search suggestion system 
        using optimized Trie data structures.
        
        ### 📚 Dataset Sizes Available
        - **1,000 words** - Fast loading, good for quick demos
        - **10,000 words** - Medium dataset, comprehensive testing
        - **100,000 words** - Large scale, production-ready demonstration
        
        ### 🔬 Algorithms Implemented
        
        **1. Naive Search - O(n×m)**
        - Linear scan through all words
        - Baseline for comparison
        - Time complexity depends on dataset size
        
        **2. Standard Trie - O(p)**
        - Classic prefix tree implementation
        - Fast searches independent of dataset size
        - Higher memory usage (26 pointers per node)
        
        **3. Compressed Trie (Patricia) - O(p)**
        - Path compression optimization
        - 60-70% reduction in node count
        - Same time complexity, better space
        
        **4. Ternary Search Trie (TST) - O(p+log n)**
        - 3-way branching instead of 26-way
        - 89% memory savings per node
        - Slight time trade-off for massive space savings
        """)
    
    with col2:
        st.markdown("""
        ### ✨ Key Features
        - Real-time autocomplete suggestions
        - Frequency-based ranking with color coding
        - Dynamic updates (learning from user selections)
        - Side-by-side algorithm comparison
        - Interactive visualizations
        - Comprehensive benchmarking
        - **Scales from 1K to 100K+ words!**
        
        ### 📊 Performance Highlights
        """)
        
        # Load and display benchmark results if available
        benchmark_data = load_benchmark_results()
        if benchmark_data and '10000' in benchmark_data.get('summary', {}):
            data_10k = benchmark_data['summary']['10000']
            
            if 'Naive' in data_10k and 'Standard Trie' in data_10k:
                naive_time = data_10k['Naive']['avg_time_ms']
                trie_time = data_10k['Standard Trie']['avg_time_ms']
                speedup = naive_time / trie_time if trie_time > 0 else 0
                
                st.metric(
                    "Speedup at 10K words",
                    f"{speedup:.2f}x faster",
                    "Standard Trie vs Naive"
                )
            
            if 'Compressed Trie' in data_10k:
                st.metric(
                    "Compressed Trie Savings",
                    data_10k['Compressed Trie']['memory_info']
                )
            
            if 'TST' in data_10k:
                st.metric(
                    "TST Memory Efficiency",
                    data_10k['TST']['memory_info']
                )
        
        st.markdown("""
        ### 🎓 Real-World Applications
        - Search engines (Google, Bing)
        - E-commerce autocomplete (Amazon)
        - IDE code completion (VS Code)
        - Mobile keyboard predictions
        - Command-line interfaces
        - DNS lookup systems
        - Spell checkers
        """)
    
    # Comparison table
    st.markdown("---")
    st.markdown("### 📋 Algorithm Comparison Table")
    
    comparison_df = pd.DataFrame({
        'Algorithm': ['Naive', 'Standard Trie', 'Compressed Trie', 'TST'],
        'Time Complexity': ['O(n×m)', 'O(p)', 'O(p)', 'O(p+log n)'],
        'Space Complexity': ['O(n×m)', 'O(26×N×M)', 'O(N×M) - 70% saved', 'O(3×N×M) - 89% saved'],
        'Best For': [
            'Small datasets',
            'Speed-critical apps',
            'Long common prefixes',
            'Memory-constrained environments'
        ],
        'Trade-off': [
            'Slow on large data',
            'High memory usage',
            'Slightly complex insertion',
            'Slightly slower than standard'
        ]
    })
    
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# ============================================================================
# TAB 2: LIVE SEARCH COMPARISON
# ============================================================================
with tab2:
    st.header("🔍 Live Search - All 4 Algorithms")
    st.markdown("### Compare performance of all algorithms in real-time")
    
    if not st.session_state.data_loaded:
        st.info("👈 Click 'Load Sample Data' in sidebar to try the demo!")
    else:
        # Google-style centered search bar
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            search_query = st.text_input(
                "Search",
                placeholder="🔍 Start typing... (try 'app', 'pro', 'com', 'test')",
                key="search_comparison",
                label_visibility="collapsed"
            )
        
        max_results = st.slider("Max results per algorithm:", 1, 10, 5)
        
        if search_query:
            st.markdown("---")
            st.markdown(f"### Results for: '{search_query}'")
            
            # Create 4 columns for 4 algorithms
            col1, col2, col3, col4 = st.columns(4)
            
            # Naive Search
            with col1:
                st.markdown("#### 🐌 Naive")
                start = time.perf_counter()
                naive_results = st.session_state.naive.search(search_query, max_results)
                naive_time = (time.perf_counter() - start) * 1000
                
                display_algo_card("Naive", naive_time, "algo-card-naive", "O(n×m)")
                
                if naive_results:
                    for idx, r in enumerate(naive_results, 1):
                        st.markdown(f"{idx}. **{r['word']}**")
                        st.caption(f"Freq: {r['frequency']}")
                else:
                    st.warning("No results")
            
            # Standard Trie
            with col2:
                st.markdown("#### ⚡ Standard Trie")
                start = time.perf_counter()
                trie_results = st.session_state.standard_trie.search(search_query, max_results)
                trie_time = (time.perf_counter() - start) * 1000
                
                speedup = naive_time / trie_time if trie_time > 0 else 0
                display_algo_card("Standard Trie", trie_time, "algo-card-standard", 
                                f"O(p) | {speedup:.1f}x faster")
                
                if trie_results:
                    for idx, r in enumerate(trie_results, 1):
                        emoji = {"green": "🟢", "yellow": "🟡", "orange": "🟠"}[r.color]
                        st.markdown(f"{idx}. {emoji} **{r.word}**")
                        st.caption(f"Freq: {r.frequency}")
                else:
                    st.warning("No results")
            
            # Compressed Trie
            with col3:
                st.markdown("#### 🎯 Compressed")
                start = time.perf_counter()
                comp_results = st.session_state.compressed_trie.search(search_query, max_results)
                comp_time = (time.perf_counter() - start) * 1000
                
                speedup = naive_time / comp_time if comp_time > 0 else 0
                display_algo_card("Compressed", comp_time, "algo-card-compressed",
                                f"O(p) | {speedup:.1f}x | 70% space saved")
                
                if comp_results:
                    for idx, r in enumerate(comp_results, 1):
                        emoji = {"green": "🟢", "yellow": "🟡", "orange": "🟠"}[r.color]
                        st.markdown(f"{idx}. {emoji} **{r.word}**")
                        st.caption(f"Freq: {r.frequency}")
                else:
                    st.warning("No results")
            
            # TST
            with col4:
                st.markdown("#### 💾 TST")
                start = time.perf_counter()
                tst_results = st.session_state.tst.search(search_query, max_results)
                tst_time = (time.perf_counter() - start) * 1000
                
                speedup = naive_time / tst_time if tst_time > 0 else 0
                display_algo_card("TST", tst_time, "algo-card-tst",
                                f"O(p+log n) | {speedup:.1f}x | 89% mem saved")
                
                if tst_results:
                    for idx, r in enumerate(tst_results, 1):
                        emoji = {"green": "🟢", "yellow": "🟡", "orange": "🟠"}[r.color]
                        st.markdown(f"{idx}. {emoji} **{r.word}**")
                        st.caption(f"Freq: {r.frequency}")
                else:
                    st.warning("No results")
            
            # Winner and comparison chart
            st.markdown("---")
            
            times = [
                ("Naive", naive_time),
                ("Standard Trie", trie_time),
                ("Compressed Trie", comp_time),
                ("TST", tst_time)
            ]
            winner = min(times, key=lambda x: x[1])
            
            col_a, col_b = st.columns([1, 2])
            
            with col_a:
                st.success(f"### 🏆 Fastest: {winner[0]}")
                st.metric("Time", f"{winner[1]:.3f} ms")
                
                avg_trie_time = (trie_time + comp_time + tst_time) / 3
                overall_speedup = naive_time / avg_trie_time if avg_trie_time > 0 else 0
                st.metric("Avg Trie Speedup", f"{overall_speedup:.2f}x")
            
            with col_b:
                # Comparison chart
                fig = create_comparison_chart(naive_time, trie_time, comp_time, tst_time)
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            # Empty state
            st.markdown("<br><br>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.info("👆 **Start typing** to compare all 4 algorithms side-by-side!")
                
                st.markdown("---")
                st.markdown("### 🎯 Try These Searches:")
                
                try_col1, try_col2 = st.columns(2)
                with try_col1:
                    st.code("app")
                    st.code("pro")
                    st.code("com")
                with try_col2:
                    st.code("test")
                    st.code("data")
                    st.code("python")

# ============================================================================
# TAB 3: TRIE VISUALIZATION
# ============================================================================
with tab3:
    st.header("🌳 Trie Structure Visualization")
    st.markdown("See how the Trie traverses to find your search prefix")
    
    if not st.session_state.data_loaded:
        st.info("👈 Load sample data first!")
    else:
        # Input for visualization
        viz_prefix = st.text_input(
            "Enter prefix to visualize:",
            placeholder="e.g., 'app'",
            key="viz_input"
        )
        
        if viz_prefix:
            viz_data = st.session_state.standard_trie.get_visualization_data(viz_prefix)
            
            if viz_data['found']:
                st.success(f"✅ Found path for '{viz_prefix}'")
                
                # Display path information
                st.markdown("### 🛤️ Traversal Path")
                
                path_str = " → ".join([node['char'] for node in viz_data['path']])
                st.markdown(f"**Root → {path_str}**")
                
                # Display each node in the path
                st.markdown("---")
                st.markdown("### 📍 Node Details")
                
                for idx, node in enumerate(viz_data['path']):
                    with st.expander(f"Node {idx + 1}: '{node['char']}' (Prefix: '{node['prefix']}')", expanded=(idx == len(viz_data['path'])-1)):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Is Word?", "✅ Yes" if node['is_word'] else "❌ No")
                        with col2:
                            st.metric("Children", node['children_count'])
                        with col3:
                            if node['is_word']:
                                st.metric("Frequency", node['frequency'])
                        
                        # Top suggestions at this node
                        if node['top_suggestions']:
                            st.markdown("**Top Suggestions:**")
                            for sugg in node['top_suggestions'][:5]:
                                st.markdown(f"- {sugg['word']} (freq: {sugg['freq']})")
                
                # Visual tree representation
                st.markdown("---")
                st.markdown("### 🌳 Tree Structure")
                
                tree_html = "<div style='font-family: monospace; line-height: 2;'>"
                tree_html += "Root<br>"
                
                indent = 0
                for node in viz_data['path']:
                    indent += 2
                    tree_html += "&nbsp;" * indent + f"└─ <b>{node['char']}</b>"
                    if node['is_word']:
                        tree_html += f" <span style='color: green;'>✓ (word)</span>"
                    tree_html += f" <span style='color: blue;'>[{node['children_count']} children]</span><br>"
                
                tree_html += "</div>"
                st.markdown(tree_html, unsafe_allow_html=True)
                
                # Complexity info
                st.markdown("---")
                st.info(f"""
                **Time Complexity:** O({len(viz_prefix)})
                
                The search only traversed {len(viz_data['path'])} nodes - 
                independent of total dataset size ({st.session_state.standard_trie.get_stats()['total_words']:,} words)!
                
                This is why Tries are perfect for autocomplete!
                """)
            else:
                st.error(f"❌ No path found for '{viz_prefix}'")
                st.info("Try: 'app', 'pro', 'com', 'test', 'data'")
        else:
            st.info("👆 Enter a prefix to visualize the Trie traversal")
            
            # Example visualization
            st.markdown("### 📚 How It Works:")
            st.markdown("""
            1. **Root Node**: Starting point of the Trie
            2. **Character Nodes**: Each character in the prefix
            3. **Word Endings**: Nodes marked as complete words
            4. **Top Suggestions**: Most frequent words at each node
            
            **The key advantage:** Trie structure allows **O(p)** time complexity 
            where p is the prefix length, completely independent of the dataset size!
            
            This means searching in 100 words takes the same time as searching 
            in 1,000,000 words!
            """)

# ============================================================================
# TAB 4: BENCHMARKS
# ============================================================================
with tab4:
    st.header("📊 Benchmark Results")
    st.markdown("### Performance analysis across different dataset sizes")
    
    benchmark_data = load_benchmark_results()
    
    if not benchmark_data:
        st.warning("""
        ⚠️ No benchmark results found. 
        
        Run benchmarks first:
        ```bash
        python benchmark.py --datasets 1k 10k 100k --iterations 30
        ```
        """)
    else:
        summary = benchmark_data.get('summary', {})
        
        # Dataset selector
        dataset_size = st.selectbox(
            "Select Dataset Size",
            options=list(summary.keys()),
            format_func=lambda x: f"{int(x):,} words"
        )
        
        if dataset_size in summary:
            data = summary[dataset_size]
            
            # Performance comparison table
            st.markdown("### 📋 Performance Comparison")
            
            table_data = []
            for algo in ['Naive', 'Standard Trie', 'Compressed Trie', 'TST']:
                if algo in data:
                    table_data.append({
                        'Algorithm': algo,
                        'Avg Time (ms)': f"{data[algo]['avg_time_ms']:.4f}",
                        'Nodes': f"{data[algo]['memory_nodes']:,}",
                        'Memory Info': data[algo]['memory_info']
                    })
            
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Speedup calculations
            st.markdown("---")
            st.markdown("### 🚀 Speedup Factors")
            
            if 'Naive' in data:
                naive_time = data['Naive']['avg_time_ms']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if 'Standard Trie' in data:
                        speedup = naive_time / data['Standard Trie']['avg_time_ms']
                        st.metric("Standard Trie", f"{speedup:.2f}x", "faster than Naive")
                
                with col2:
                    if 'Compressed Trie' in data:
                        speedup = naive_time / data['Compressed Trie']['avg_time_ms']
                        st.metric("Compressed Trie", f"{speedup:.2f}x", 
                                f"faster + {data['Compressed Trie']['memory_info'].split('(')[1].split(')')[0] if '(' in data['Compressed Trie']['memory_info'] else '70% saved'}")
                
                with col3:
                    if 'TST' in data:
                        speedup = naive_time / data['TST']['avg_time_ms']
                        st.metric("TST", f"{speedup:.2f}x", 
                                f"faster + {data['TST']['memory_info'].split('(')[1].split(')')[0] if '(' in data['TST']['memory_info'] else '89% saved'}")
        
        # Overall performance charts
        st.markdown("---")
        st.markdown("### 📈 Performance Across Dataset Sizes")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = create_benchmark_performance_chart(benchmark_data)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = create_speedup_chart(benchmark_data)
            st.plotly_chart(fig2, use_container_width=True)
        
        # Key insights
        st.markdown("---")
        st.markdown("### 🎯 Key Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Algorithm Strengths:**
            
            - **Naive:** Simple, but O(n) scales poorly
            - **Standard Trie:** Fastest, but memory intensive
            - **Compressed Trie:** Best balance for long prefixes
            - **TST:** Best for memory-constrained systems
            
            **When to use what:**
            - Small data (<1K): Naive is fine
            - Speed critical: Standard Trie
            - Balanced needs: Compressed Trie
            - Limited memory: TST
            """)
        
        with col2:
            st.markdown("""
            **Performance Scaling:**
            
            - **Naive:** Linear O(n) - doubles with dataset
            - **Tries:** Constant O(p) - independent of size
            
            **Real Impact:**
            At 100K words, Tries are **100-200x faster** than Naive!
            
            **Space-Time Tradeoff:**
            - Standard Trie: Speed ↑↑, Memory ↓
            - Compressed: Speed ↑, Memory ↑
            - TST: Speed ↑, Memory ↑↑
            """)

# ============================================================================
# TAB 5: DASHBOARD
# ============================================================================
with tab5:
    st.header("📊 Analytics Dashboard")
    st.markdown("### Word statistics and system metrics")
    
    if not st.session_state.data_loaded:
        st.info("👈 Load sample data first!")
    else:
        # Word Statistics Table
        st.subheader("📋 Word Statistics")
        
        all_words = st.session_state.standard_trie.get_all_words()
        if all_words:
            df = pd.DataFrame(all_words)
            
            # Controls
            col1, col2, col3 = st.columns(3)
            
            with col1:
                sort_by = st.selectbox("Sort by:", 
                    ["Frequency (High to Low)", "Frequency (Low to High)", "Alphabetical"])
            
            with col2:
                filter_color = st.selectbox("Filter by popularity:", 
                    ["All", "Green (Popular)", "Yellow (Moderate)", "Orange (Less)"])
            
            with col3:
                show_count = st.slider("Show top N words:", 10, len(all_words), min(50, len(all_words)))
            
            # Apply filters
            if filter_color != "All":
                color_map = {
                    "Green (Popular)": "green",
                    "Yellow (Moderate)": "yellow",
                    "Orange (Less)": "orange"
                }
                df = df[df['color'] == color_map[filter_color]]
            
            # Apply sorting
            if sort_by == "Frequency (High to Low)":
                df = df.sort_values('frequency', ascending=False)
            elif sort_by == "Frequency (Low to High)":
                df = df.sort_values('frequency', ascending=True)
            else:
                df = df.sort_values('word')
            
            # Show top N
            df = df.head(show_count)
            
            # Color code
            def color_code(row):
                colors = {'green': "#00560B", 'yellow': "#866806", 'orange': "#984400"}
                return [f'background-color: {colors.get(row["color"], "#ffffff")}'] * len(row)
            
            st.dataframe(
                df.style.apply(color_code, axis=1),
                use_container_width=True,
                height=400
            )
            
            # Export button
            if st.button("📥 Export to CSV"):
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    "word_statistics.csv",
                    "text/csv",
                    key='download-csv'
                )
        
        st.markdown("---")
        
        # System metrics comparison
        st.subheader("🔧 System Metrics - All Algorithms")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Standard Trie")
            standard_stats = st.session_state.standard_trie.get_stats()
            for key, value in standard_stats.items():
                st.text(f"{key}: {value}")
        
        with col2:
            st.markdown("#### Compressed Trie")
            comp_stats = st.session_state.compressed_trie.get_stats()
            for key, value in comp_stats.items():
                st.text(f"{key}: {value}")
        
        with col3:
            st.markdown("#### TST")
            tst_stats = st.session_state.tst.get_stats()
            for key, value in tst_stats.items():
                st.text(f"{key}: {value}")
        
        # Frequency distribution
        st.markdown("---")
        st.subheader("📊 Frequency Distribution")
        
        if all_words:
            df = pd.DataFrame(all_words)
            
            fig = px.histogram(
                df, 
                x='frequency', 
                nbins=30,
                color='color',
                color_discrete_map={'green': '#28a745', 'yellow': '#ffc107', 'orange': '#fd7e14'},
                title="Word Frequency Distribution"
            )
            
            fig.update_layout(
                xaxis_title="Frequency",
                yaxis_title="Number of Words",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><b>🔍 Trie Search Optimization Project - Complete 4 Algorithm Comparison</b></p>
    <p>Built with Python, Streamlit, and Advanced Data Structures</p>
    <p>Demonstrates: Naive Search, Standard Trie, Compressed Trie (Patricia), and Ternary Search Trie (TST)</p>
</div>
""", unsafe_allow_html=True)