import streamlit as st
import os
import re
import random
import xml.etree.ElementTree as ET
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
DATASET_FOLDER = "/Users/dhritichandan/Downloads/SFILES_2/Dhriti_Test/new_dataset/graphml_10k_dataset"
MODEL_FILENAME = "transition_model.pkl"

MODEL_ORDER = 2
NUM_TOP_PATHS_TO_FIND = 5
NUM_PREDICTION_OPTIONS = 5
MAX_NEIGHBORS_TO_SAMPLE = 75

# --- CORE LOGIC (Cached for Performance) ---

@st.cache_data
def get_node_type(node_id):
    """Generalizes a node ID like 'hex1' or 'pp-123' back to its base type 'hex' or 'pp'."""
    base = re.sub(r'\d+$', '', node_id)
    if base.endswith('-'):
        return base[:-1]
    return base

def load_graphml(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except ET.ParseError: return {}, []
    ns = {"graphml": "http://graphml.graphdrawing.org/xmlns"}
    nodes, edges = {}, []
    for node in root.findall(".//graphml:node", ns):
        nodes[node.attrib["id"]] = ""
    for edge in root.findall(".//graphml:edge", ns):
        edges.append((edge.attrib["source"], edge.attrib["target"]))
    return nodes, edges

@st.cache_resource(show_spinner="Building main graph from dataset...")
def build_graph(dataset_folder):
    if not os.path.isdir(dataset_folder):
        st.error(f"Dataset folder not found at: {dataset_folder}")
        st.stop()
        
    outgoing_graph = defaultdict(list)
    for filename in os.listdir(dataset_folder):
        if filename.startswith("pid") and filename.endswith(".graphml"):
            file_path = os.path.join(dataset_folder, filename)
            _, edges = load_graphml(file_path)
            for src, tgt in edges:
                outgoing_graph[src].append(tgt)
    return outgoing_graph

@st.cache_resource(show_spinner="Building reversed graph...")
def build_reversed_graph(_outgoing_graph):
    incoming_graph = defaultdict(list)
    for node, neighbors in _outgoing_graph.items():
        for neighbor in neighbors:
            incoming_graph[neighbor].append(node)
    return incoming_graph

@st.cache_resource(show_spinner="Loading or Creating Prediction Model...")
def get_or_create_transition_model(_outgoing_graph, filename=MODEL_FILENAME):
    if os.path.exists(filename):
        st.info(f"Loading pre-computed model from '{filename}'...")
        with open(filename, "rb") as f:
            model_dict = pickle.load(f)
        transitions = defaultdict(Counter)
        for context, counts in model_dict.items():
            transitions[context] = Counter(counts)
        st.success("Model loaded successfully!")
        return transitions

    st.warning("First-time setup: Pre-computing prediction model. This might take several minutes...")
    with st.spinner("Analyzing graph and building model... Please wait."):
        transitions = defaultdict(Counter)
        graph_items = list(_outgoing_graph.items())
        for node_a, neighbors_of_a in graph_items:
            if not neighbors_of_a: continue
            sampled_neighbors_a = random.sample(neighbors_of_a, min(len(neighbors_of_a), MAX_NEIGHBORS_TO_SAMPLE))
            for node_b in sampled_neighbors_a:
                neighbors_of_b = _outgoing_graph.get(node_b, [])
                if not neighbors_of_b: continue
                sampled_neighbors_b = random.sample(neighbors_of_b, min(len(neighbors_of_b), MAX_NEIGHBORS_TO_SAMPLE))
                context = (get_node_type(node_a), get_node_type(node_b))
                for node_c in sampled_neighbors_b:
                    next_node = get_node_type(node_c)
                    transitions[context][next_node] += 1
    
    st.info(f"Saving model to '{filename}' for fast startup next time...")
    model_to_save = {k: dict(v) for k, v in transitions.items()}
    with open(filename, "wb") as f:
        pickle.dump(model_to_save, f)
    st.success("Model built and saved! The app will now load instantly.")
    return transitions

@st.cache_data(show_spinner="Finding common paths to pumps...")
def find_top_paths_to_pump(_graph, _reversed_graph, pump_keyword="pp", top_n=5, max_len=4):
    in_degree_counts = Counter(neighbor for neighbors in _graph.values() for neighbor in neighbors)
    pump_type = get_node_type(pump_keyword)
    direct_predecessors = [node for node, neighbors in _graph.items() for neighbor in neighbors if pump_keyword in neighbor.lower()]
    if not direct_predecessors: return []
    top_predecessors = [node for node, _ in Counter(direct_predecessors).most_common(top_n)]
    top_paths = []
    for start_node in top_predecessors:
        path = [start_node]
        while len(path) < max_len - 1:
            head_node = path[0]
            predecessors = _reversed_graph.get(head_node, [])
            if not predecessors: break
            best_predecessor = max(predecessors, key=lambda p: in_degree_counts.get(p, 0))
            if best_predecessor in path: break
            path.insert(0, best_predecessor)
        generalized_path = tuple(get_node_type(n) for n in path) + (pump_type,)
        top_paths.append(generalized_path)
    unique_paths = list(dict.fromkeys(top_paths))
    return unique_paths

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

# <<< REVERTED: This function now uses the graphviz library directly >>>
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

outgoing_graph = build_graph(DATASET_FOLDER)
reversed_graph = build_reversed_graph(outgoing_graph)
transition_model = get_or_create_transition_model(outgoing_graph)
pump_sequences = find_top_paths_to_pump(outgoing_graph, reversed_graph, top_n=NUM_TOP_PATHS_TO_FIND)

if 'building' not in st.session_state:
    st.session_state.building = False
    st.session_state.active_path = []
    st.session_state.constructed_graph = defaultdict(list)
    st.session_state.suggested_path = None

if not st.session_state.building:
    st.header("1. Choose a common path leading to a pump")
    if not pump_sequences:
        st.warning("Could not find any common paths to a pump in the dataset.")
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
        
        # <<< REVERTED: Call the original render_graph function >>>
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
                for key in list(st.session_state.keys()): del st.session_state.key
                st.rerun()
