import streamlit as st
import os
import re
from collections import defaultdict, Counter
import graphviz
import pickle

# --- App Configuration ---
st.set_page_config(
    page_title="Interactive P&ID Path Builder",
    page_icon="⚗️",
    layout="wide"
)

# --- GLOBAL VARIABLES & CONSTANTS ---
MODEL_FILENAME = "bayesian_data.pkl"
MODEL_ORDER = 2
NUM_PREDICTION_OPTIONS = 5

# --- CORE LOGIC (Cached for Performance) ---

@st.cache_data
def get_node_type(node_id):
    """
    Generalizes a node ID like 'hex1', 'pp-123', or 'c-2/m' back to 
    its base type 'hex', 'pp', or 'c/m'.
    """
    # First, remove any hyphen followed by digits (e.g., 'c-2/m' -> 'c/m')
    base = re.sub(r'-\d+', '', node_id)
    # Then, remove any trailing digits (e.g., 'hex1' -> 'hex')
    base = re.sub(r'\d+$', '', base)
    # Final cleanup for any remaining trailing hyphens
    if base.endswith('-'):
        return base[:-1]
    return base

@st.cache_resource(show_spinner="Loading application data...")
def load_app_data(filename=MODEL_FILENAME):
    """Loads the pre-computed model and starting paths from the .pkl file."""
    if not os.path.exists(filename):
        st.error(f"Model file '{filename}' not found!")
        st.info("This file should be in your GitHub repository. Please ensure the 'precompute.py' script was run and 'app_data.pkl' was uploaded correctly using Git LFS.")
        st.stop()
    
    with open(filename, "rb") as f:
        app_data = pickle.load(f)
        
    # Reconstruct the Counter objects for the model
    transitions = defaultdict(Counter)
    for context, counts in app_data["transition_model"].items():
        transitions[context] = Counter(counts)
        
    return transitions, app_data["pump_sequences"]

# --- VISUALIZATION & HELPER FUNCTIONS ---
def get_unique_node_name(base_name, existing_nodes):
    if base_name not in existing_nodes:
        return base_name
    i = 1
    while True:
        new_name = f"{base_name}{i}"
        if new_name not in existing_nodes:
            return new_name
        i += 1

def render_graph(constructed_graph, active_node=None):
    dot = graphviz.Digraph('P&ID', comment='Process Flow Diagram')
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
    dot.attr('edge', color='gray40')
    dot.attr(rankdir='LR')

    if not constructed_graph: return dot

    all_nodes = set(constructed_graph.keys())
    for children in constructed_graph.values(): all_nodes.update(children)
    
    for node in all_nodes:
        dot.node(node, node, fillcolor='mediumseagreen' if node == active_node else 'lightblue')
        
    for parent, children in constructed_graph.items():
        for child in children:
            dot.edge(parent, child)
    return dot

def find_most_probable_path(start_path, model, max_len=5):
    path = list(start_path)
    path_types_tracker = {get_node_type(n) for n in path}
    
    for _ in range(max_len):
        if len(path) < MODEL_ORDER: break
        
        context = tuple([get_node_type(node) for node in path[-MODEL_ORDER:]])
        if context not in model: break

        options = model[context].most_common(1)
        if not options: break
            
        next_node_type = options[0][0]
        if next_node_type in path_types_tracker:
            path.append(f"[Loop to {next_node_type}]")
            break

        path.append(next_node_type)
        path_types_tracker.add(next_node_type)
        
    return path[len(start_path):]

# --- STREAMLIT APP UI & LOGIC ---

st.title("⚗️ Interactive P&ID Path Builder")
st.write("A tool to interactively build and traverse Process & Instrumentation Diagrams.")

# Load the pre-computed data. The app's functionality starts here.
transition_model, pump_sequences = load_app_data()

if 'building' not in st.session_state:
    st.session_state.building = False
    st.session_state.active_path = []
    st.session_state.constructed_graph = defaultdict(list)
    st.session_state.suggested_path = None

if not st.session_state.building:
    st.header("1. Choose a common path leading to a pump")
    if not pump_sequences:
        st.warning("Could not find any common paths to a pump in the pre-computed data.")
    else:
        path_options = {f"{' -> '.join(seq)}": seq for seq in pump_sequences}
        chosen_path_str = st.selectbox("Select a starting sequence:", options=path_options.keys())
        
        if st.button("Start Building Path", type="primary"):
            initial_sequence_generic = list(path_options[chosen_path_str])
            
            temp_graph = defaultdict(list)
            all_nodes = set()
            path_so_far_unique = []
            for node_type in initial_sequence_generic:
                unique_name = get_unique_node_name(node_type, all_nodes)
                all_nodes.add(unique_name)
                path_so_far_unique.append(unique_name)
            
            st.session_state.active_path = path_so_far_unique
            for i in range(len(path_so_far_unique) - 1):
                temp_graph[path_so_far_unique[i]].append(path_so_far_unique[i+1])
            st.session_state.constructed_graph = temp_graph
            
            st.session_state.building = True
            st.rerun()
else:
    st.header("2. Build Your P&ID")

    col1, col2 = st.columns([0.6, 0.4])

    with col1:
        st.subheader("Live P&ID Graph")
        active_node = st.session_state.active_path[-1] if st.session_state.active_path else None
        st.write("The green node is your current position. Use the controls to navigate or build.")
        graph_viz = render_graph(st.session_state.constructed_graph, active_node)
        st.graphviz_chart(graph_viz)

    with col2:
        st.subheader("Controls")
        st.info(f"**Current Path:** `{' -> '.join(st.session_state.active_path)}`")
        
        current_node = st.session_state.active_path[-1]
        existing_children = st.session_state.constructed_graph.get(current_node, [])

        st.write("---")
        
        if existing_children:
            st.write("**Navigate to an existing branch:**")
            selected_child = st.selectbox("Follow branch to:", options=existing_children, index=None, placeholder="Choose a destination...")
            if st.button("Follow Branch", disabled=(selected_child is None)):
                st.session_state.active_path.append(selected_child)
                st.session_state.suggested_path = None
                st.rerun()
            st.write("---")

        st.write("**Let the AI suggest the next steps:**")
        if st.button("Suggest Most Probable Path"):
            suggested = find_most_probable_path(st.session_state.active_path, transition_model)
            st.session_state.suggested_path = suggested if suggested else ["No probable path found."]
        
        if st.session_state.suggested_path:
            st.write("Suggested path:")
            st.info(f"`{current_node} -> {' -> '.join(st.session_state.suggested_path)}`")
            if st.button("Add This Path to Diagram", type="primary"):
                path_to_add = st.session_state.suggested_path
                parent = current_node
                
                all_nodes = set(st.session_state.constructed_graph.keys())
                for children in st.session_state.constructed_graph.values(): all_nodes.update(children)

                for i, node_type in enumerate(path_to_add):
                    if "[Loop" in node_type: continue
                    unique_name = get_unique_node_name(node_type, all_nodes)
                    st.session_state.constructed_graph[parent].append(unique_name)
                    st.session_state.active_path.append(unique_name)
                    all_nodes.add(unique_name)
                    parent = unique_name
                
                st.session_state.suggested_path = None
                st.rerun()
        st.write("---")
        
        options = []
        if len(st.session_state.active_path) >= MODEL_ORDER:
            context = tuple([get_node_type(node) for node in st.session_state.active_path[-MODEL_ORDER:]])
            if context in transition_model:
                options_with_counts = transition_model[context].most_common(NUM_PREDICTION_OPTIONS)
                total = sum(count for _, count in options_with_counts)
                if total > 0:
                    options = [f"{node} ({ (count/total)*100 :.1f}%)" for node, count in options_with_counts]
        
        if options:
            st.write("**Or, add a new branch manually:**")
            selected_options = st.multiselect("Select nodes to add:", options, label_visibility="collapsed")
            if st.button("Add Selected Node(s)", disabled=not selected_options):
                all_nodes = set(st.session_state.constructed_graph.keys())
                for children in st.session_state.constructed_graph.values(): all_nodes.update(children)
                
                newly_added_nodes = []
                for option_str in selected_options:
                    base_node_type = option_str.split(' ')[0]
                    unique_name = get_unique_node_name(base_node_type, all_nodes)
                    st.session_state.constructed_graph[current_node].append(unique_name)
                    newly_added_nodes.append(unique_name)
                    all_nodes.add(unique_name)
                
                if newly_added_nodes:
                    st.session_state.active_path.append(newly_added_nodes[-1])
                st.session_state.suggested_path = None
                st.rerun()
        else:
             st.success("Path complete! No further probable transitions found for manual adding.")


        st.write("---")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Go Back One Step", use_container_width=True):
                if len(st.session_state.active_path) > 1:
                    st.session_state.active_path.pop()
                    st.session_state.suggested_path = None
                    st.rerun()
                else:
                    st.warning("Cannot go back further.")
        with c2:
            if st.button("Finish & Restart", use_container_width=True):
                for key in list(st.session_state.keys()): del st.session_state[key]
                st.rerun()