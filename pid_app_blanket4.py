import os
import re
import random
import pickle
from collections import defaultdict, Counter
import streamlit as st
import pandas as pd

# -------- SETTINGS --------
MODEL_ORDER = 2
NUM_PREDICTION_OPTIONS = 5
MODEL_FILE = "pid_models.pkl"  # This must match the output of preprocess_data.py

# -------- Helper Functions --------
def get_node_type(node_id):
    """Generalizes a node ID by removing numbers/separators to find its type."""
    clean_id = re.sub(r'[_\-\d]+', '', node_id)
    return clean_id if clean_id else "unknown"

def get_unique_id(base_type, existing_nodes):
    """Generates a guaranteed unique ID (e.g., Pump_1, Pump_2)."""
    clean_base = get_node_type(base_type)
    counter = 1
    new_id = f"{clean_base}_{counter}"
    while new_id in existing_nodes:
        counter += 1
        new_id = f"{clean_base}_{counter}"
    return new_id

def graph_to_graphviz(nodes, edges, focus_node=None):
    """Converts node/edge lists into a Graphviz DOT language string handling branches."""
    if not nodes:
        return 'digraph {}'
    
    dot_string = 'digraph G {\n'
    dot_string += '  rankdir=LR;\n'
    dot_string += '  node [shape=box, style="rounded,filled", fillcolor=skyblue, fontname="Helvetica"];\n'
    dot_string += '  edge [fontname="Helvetica"];\n'

    # Add Nodes
    for n in nodes:
        color = 'gold' if n == focus_node else 'skyblue'
        if 'Start' in n: color = 'palegreen'
        if 'Splitter' in n or 'splt' in n.lower(): 
            dot_string += f'  "{n}" [shape=diamond, fillcolor={color}];\n'
        else:
            dot_string += f'  "{n}" [fillcolor={color}];\n'

    # Add Edges
    for source, target in edges:
        dot_string += f'  "{source}" -> "{target}";\n'
    
    dot_string += '}'
    return dot_string

def get_markov_blanket_stats(node_type, blanket_model):
    """Retrieves statistical norms for a specific node type."""
    if node_type in blanket_model:
        return blanket_model[node_type]
    return {"parents": Counter(), "children": Counter(), "spouses": Counter()}

def generate_demo_data():
    """Generates synthetic 'Dummy' data including Splitter logic."""
    transitions = defaultdict(Counter)
    blanket_model = defaultdict(lambda: {"parents": Counter(), "children": Counter(), "spouses": Counter()})
    
    # --- Blanket Stats (Graph Rules) ---
    # Raw: Source node
    blanket_model['Raw']['children']['Pump'] = 15
    blanket_model['Raw']['parents']['Start'] = 10
    
    # Splitter: 1 Input -> Many Outputs
    blanket_model['Splitter']['children']['Valve'] = 20
    blanket_model['Splitter']['children']['Sensor'] = 15
    blanket_model['Splitter']['parents']['Pump'] = 10
    
    # Pump: 1 Input -> 1 Output (Safety)
    blanket_model['Pump']['children']['CheckValve'] = 25 
    blanket_model['Pump']['parents']['Raw'] = 15
    
    # Mixer: Many Inputs -> 1 Output
    blanket_model['Mixer']['parents']['Valve'] = 10
    blanket_model['Mixer']['parents']['Heater'] = 10
    blanket_model['Mixer']['children']['Tank'] = 10

    top_paths = [['Start', 'Raw'], ['Start', 'Pump']]
    return transitions, blanket_model, top_paths

# -------- Core Loading Logic --------
@st.cache_data
def load_precomputed_models(filepath):
    """Loads the processed pickle file."""
    if not os.path.exists(filepath):
        return None, None, None
    
    try:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        return data["transitions"], data["blanket_model"], data["top_paths"]
    except Exception as e:
        st.error(f"Error loading model file: {e}")
        return None, None, None

# -------- Main Streamlit App Logic --------
st.set_page_config(layout="wide", page_title="PID Smart Builder")
st.title(" Intelligent PID Graph Builder")
st.markdown("""
This tool uses **Markov Blankets (Graph Logic)**.
It predicts connections by analyzing the structural requirements of the active component (Inputs/Outputs/Parallelism).
""")

# --- Sidebar ---
st.sidebar.header("Configuration")
st.sidebar.info(f"Looking for model file: `{MODEL_FILE}`")

if st.sidebar.button("Refresh Models"):
    st.cache_data.clear()
    st.session_state.data_loaded = False
    st.rerun()

# --- Load Data Logic ---
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

if not st.session_state.data_loaded:
    with st.spinner("Loading AI Models from disk..."):
        trans, blanket, paths = load_precomputed_models(MODEL_FILE)
        
    if trans:
        st.session_state.transitions = trans
        st.session_state.blanket_model = blanket
        st.session_state.start_paths = paths
        st.session_state.all_node_types = sorted(list(blanket.keys()))
        st.session_state.data_loaded = True
        st.toast("Models loaded successfully!")
        st.rerun()
    else:
        st.warning(f"Using Demo Data (File `{MODEL_FILE}` not found).")
        trans, blanket, paths = generate_demo_data()
        st.session_state.transitions = trans
        st.session_state.blanket_model = blanket
        st.session_state.start_paths = paths
        st.session_state.all_node_types = sorted(list(blanket.keys()))
        st.session_state.data_loaded = True
        st.rerun()

# --- Initialize State ---
if 'stage' not in st.session_state:
    st.session_state.stage = 'start'
if 'graph_nodes' not in st.session_state:
    st.session_state.graph_nodes = []
if 'graph_edges' not in st.session_state:
    st.session_state.graph_edges = []
if 'focus_node' not in st.session_state:
    st.session_state.focus_node = None

# --- APP FLOW ---

if st.session_state.get('data_loaded', False):
    
    # === STEP 1: START ===
    if st.session_state.stage == 'start':
        st.subheader("Step 1: Start the Graph")
        
        cols = st.columns(3)
        for i, seq in enumerate(st.session_state.start_paths):
            with cols[i % 3]:
                path_str = ' → '.join(seq)
                if st.button(f"{path_str}", key=f"start_{i}", use_container_width=True):
                    st.session_state.graph_nodes = []
                    st.session_state.graph_edges = []
                    prev_node = None
                    for node_type in seq:
                        new_id = get_unique_id(node_type, st.session_state.graph_nodes)
                        st.session_state.graph_nodes.append(new_id)
                        if prev_node:
                            st.session_state.graph_edges.append((prev_node, new_id))
                        prev_node = new_id
                    st.session_state.focus_node = prev_node
                    st.session_state.stage = 'predicting'
                    st.rerun()
        
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            if st.button("Start with Custom Empty Path"):
                st.session_state.graph_nodes = ["Start"]
                st.session_state.graph_edges = []
                st.session_state.focus_node = "Start"
                st.session_state.stage = 'predicting'
                st.rerun()
        with col_c2:
            if st.button("Start with Splitter (Parallel)"):
                st.session_state.graph_nodes = ["Start", "Splitter_1"]
                st.session_state.graph_edges = [("Start", "Splitter_1")]
                st.session_state.focus_node = "Splitter_1"
                st.session_state.stage = 'predicting'
                st.rerun()
        with col_c3:
            if st.button("Start with Raw Material"):
                st.session_state.graph_nodes = ["Start", "Raw_1"]
                st.session_state.graph_edges = [("Start", "Raw_1")]
                st.session_state.focus_node = "Raw_1"
                st.session_state.stage = 'predicting'
                st.rerun()

    # === STEP 2: PREDICT & BRANCH ===
    elif st.session_state.stage == 'predicting':
        
        col_main, col_tools = st.columns([2, 1])
        
        with col_main:
            st.subheader("Interactive Graph")
            
            graph_viz = graph_to_graphviz(
                st.session_state.graph_nodes, 
                st.session_state.graph_edges, 
                st.session_state.focus_node
            )
            st.graphviz_chart(graph_viz)
            
            # Focus Node Selector
            st.write("---")
            st.write("###  Active Node")
            col_f1, col_f2 = st.columns([3,1])
            with col_f1:
                if st.session_state.focus_node not in st.session_state.graph_nodes:
                    st.session_state.focus_node = st.session_state.graph_nodes[-1]
                
                new_focus = st.selectbox(
                    "Select node to edit/extend:", 
                    st.session_state.graph_nodes, 
                    index=st.session_state.graph_nodes.index(st.session_state.focus_node)
                )
                if new_focus != st.session_state.focus_node:
                    st.session_state.focus_node = new_focus
                    st.rerun()
            
            with col_f2:
                if st.button("Finish Graph"):
                    st.session_state.stage = 'done'
                    st.rerun()

        with col_tools:
            st.subheader("Graph Suggestions")
            
            current_type = get_node_type(st.session_state.focus_node)
            blanket_stats = get_markov_blanket_stats(current_type, st.session_state.blanket_model)
            
            # Direction Selection
            direction = st.radio("Connect:", ["Next (Output)", "Prev (Input)"], horizontal=True)
            
            # --- PURE MARKOV BLANKET ENGINE (GRAPH LOGIC) ---
            options = Counter()
            
            if "Next" in direction:
                if blanket_stats['children']:
                    options = blanket_stats['children']
                    st.info(f"Based on **Outputs**: `{current_type}` usually feeds into these.")
            else:
                if blanket_stats['parents']:
                    options = blanket_stats['parents']
                    st.info(f"Based on **Inputs**: `{current_type}` usually receives flow from these.")

            source_node = st.session_state.focus_node
            
            if options:
                top_opts = [n for n, c in options.most_common(10)]
                selected_nodes = st.multiselect("Select component(s) to add:", top_opts, format_func=lambda x: f"{x}")
                
                if st.button("Attach Selected", type="primary"):
                    added_nodes = []
                    for node_type in selected_nodes:
                        new_id = get_unique_id(node_type, st.session_state.graph_nodes)
                        st.session_state.graph_nodes.append(new_id)
                        
                        # Directional Edge Logic
                        if "Next" in direction:
                            st.session_state.graph_edges.append((source_node, new_id))
                        else:
                            st.session_state.graph_edges.append((new_id, source_node))
                            
                        added_nodes.append(new_id)
                    
                    if len(added_nodes) == 1:
                         st.session_state.focus_node = added_nodes[0]
                    else:
                        st.toast(f"Branched {len(added_nodes)} paths from {source_node}")
                    st.rerun()
            else:
                st.warning("No structural data found for this component.")
            
            # --- MANUAL FALLBACK ---
            st.divider()
            with st.expander(" Manual Component Add", expanded=not options):
                manual_node = st.selectbox("Add any component:", st.session_state.all_node_types)
                if st.button("Add Manually"):
                    new_id = get_unique_id(manual_node, st.session_state.graph_nodes)
                    st.session_state.graph_nodes.append(new_id)
                    
                    # Directional Edge Logic for Manual Add
                    if "Next" in direction:
                        st.session_state.graph_edges.append((source_node, new_id))
                    else:
                        st.session_state.graph_edges.append((new_id, source_node))
                        
                    st.session_state.focus_node = new_id
                    st.rerun()

            st.divider()
            st.markdown("###  Structural Validation")

            # --- 1. OUTGOING (Missing Children) ---
            if blanket_stats['children']:
                total_out = sum(blanket_stats['children'].values())
                # If a child appears in >30% of cases, it's considered "Standard"
                standard_children = [k for k,v in blanket_stats['children'].items() if (v/total_out) > 0.3]
                
                current_children_types = [get_node_type(t) for s, t in st.session_state.graph_edges if s == st.session_state.focus_node]
                
                missing_out = [c for c in standard_children if c not in current_children_types]
                if missing_out:
                    msg = f"Missing Output: `{current_type}` usually connects to **{', '.join(missing_out)}**."
                    st.warning(f"⚠️ {msg}")

            # --- 2. INCOMING (Parallel Input Check) ---
            if blanket_stats['parents']:
                total_in = sum(blanket_stats['parents'].values())
                standard_parents = [k for k,v in blanket_stats['parents'].items() if (v/total_in) > 0.3]
                
                current_parent_types = [get_node_type(s) for s, t in st.session_state.graph_edges if t == st.session_state.focus_node]
                
                # If we have SOME input, but are missing other STANDARD inputs (Parallel requirement)
                if current_parent_types: 
                    missing_in = [p for p in standard_parents if p not in current_parent_types]
                    if missing_in:
                        msg = f"Missing Input: `{current_type}` also typically receives input from **{', '.join(missing_in)}**."
                        st.info(f"⚠️ {msg}")

    # === DONE ===
    elif st.session_state.stage == 'done':
        st.success("Graph Construction Complete")
        st.graphviz_chart(graph_to_graphviz(st.session_state.graph_nodes, st.session_state.graph_edges))
        
        if st.button("Start New Graph"):
            st.session_state.stage = 'start'
            st.session_state.graph_nodes = []
            st.session_state.graph_edges = []
            st.rerun()
