import streamlit as st
import networkx as nx
import graphviz
import json
import re
from xml import etree
import matplotlib.pyplot as plt
import io
from collections import Counter

# --- App Configuration ---
st.set_page_config(
    page_title="Advanced P&ID File Converter",
    page_icon="🛠️",
    layout="wide"
)

# --- ADVANCED LOGIC & MAPPINGS (SFILES Rules Compliant) ---

COLOR_MAP = {
    'valve': '#FF6347', 'connector': '#4682B4', 'pipe': '#32CD32', 'mixer': '#FFD700',
    'pump': '#6A5ACD', 'splitter': '#FFA500', 'vessel': '#20B2AA', 'instrument': '#8A2BE2',
    'general': '#D3D3D3', 'tank': '#ADD8E6', 'heater': '#F08080', 'reactor': '#DA70D6',
    'raw_material': '#B0C4DE', 'product': '#98FB98', 'control': '#DDA0DD',
    'DEFAULT': '#808080'
}

PREFIX_MAP = {
    'v': ('VALVE', 'Valve'), 'pp': ('PUMP', 'Pump'), 'mix': ('MIXER', 'Mixer'),
    'splt': ('SPLITTER', 'Splitter'), 'r': ('REACTOR', 'Reactor'), 'tk': ('TANK', 'Tank'),
    'hex': ('HEATER', 'Heat Exchanger'), 'prod': ('PRODUCT', 'Product Stream'), 'raw': ('RAW_MATERIAL', 'Raw Material'),
    'c': ('CONTROL', 'Control')
}

CONTROLLER_TYPE_MAP = {
    'FC': 'Flow Controller', 'LC': 'Level Controller', 'PC': 'Pressure Controller',
    'TC': 'Temperature Controller', 'FI': 'Flow Indicator', 'PI': 'Pressure Indicator',
    'M': 'Manual Controller',
    'FRC': 'Flow Ratio Controller'
}

def infer_attributes_from_id(node_id, controller_type=None):
    """
    Intelligently infers component TYPE and LABEL from its ID, now handling complex controllers.
    """
    match = re.match(r"([a-zA-Z_]+)-(\d+)(?:/([A-Z_]+))?", str(node_id))
    if not match: return {'type': 'UNKNOWN', 'label': str(node_id)}
    
    prefix, number, subtype = match.groups()
    
    final_controller_type = controller_type if controller_type else subtype

    if prefix.lower() == 'c' and final_controller_type:
        inferred_type = 'CONTROL'
        base_label = CONTROLLER_TYPE_MAP.get(final_controller_type, f"Control {final_controller_type}")
    else:
        inferred_type, base_label = PREFIX_MAP.get(prefix.lower(), ('UNKNOWN', prefix.upper()))
        
    return {'type': inferred_type, 'label': f"{base_label} {number}"}

def enrich_graph_with_inferred_attributes(G):
    for node, data in G.nodes(data=True):
        if 'type' not in data or 'label' not in data or data.get('type') is None:
            inferred_attrs = infer_attributes_from_id(node)
            data.setdefault('type', inferred_attrs['type'])
            data.setdefault('label', inferred_attrs['label'])
    return G

# --- SFILES (COMPACT SINGLE-LINE FORMAT) - NEW ADVANCED PARSER ---

def compact_sfile_to_networkx(sfile_content):
    """
    Parses the compact SFILES format according to the advanced rules
    (numbered tags, cycles, branches, controllers).
    """
    G = nx.DiGraph()
    for part in sfile_content.split('n|'):
        content = part.strip().replace('\n', '')
        token_pattern = re.compile(r'(\([a-zA-Z_]+\)(?:\{[a-zA-Z_]+\})?(?:\{\d+\})?|\[|\]|<_?\d+>|_?\d+)')
        tokens = [match.group(0) for match in token_pattern.finditer(content)]
        
        path_stack, last_node, type_counts = [], None, Counter()
        cycle_sources, signal_sources = {}, {}
        
        # This pass builds the main sequential graph and logs where loop tags are
        for token in tokens:
            if token == '[':
                if last_node: path_stack.append(last_node)
            elif token == ']':
                if path_stack: last_node = path_stack.pop()
            elif token.startswith('('):
                comp_match = re.match(r'\(([a-zA-Z_]+)\)(?:\{([a-zA-Z_]+)\})?(?:\{(\d+)\})?', token)
                comp_type, controller_type, num_str = comp_match.groups()
                node_id = f"{comp_type}-{num_str}" if num_str else f"{comp_type}-{type_counts.get(comp_type, 0) + 1}"
                if not num_str: type_counts[comp_type] += 1
                if not G.has_node(node_id):
                    attrs = infer_attributes_from_id(node_id, controller_type=controller_type)
                    G.add_node(node_id, **attrs)
                if last_node:
                    parent = path_stack[-1] if path_stack else last_node
                    G.add_edge(parent, node_id, type='process')
                last_node = node_id
            elif re.match(r'^_?\d+$', token): # Source tag
                is_signal = token.startswith('_')
                num = int(token.strip('_'))
                if is_signal: signal_sources[num] = last_node
                else: cycle_sources[num] = last_node
            elif token.startswith('<'): # Target tag
                is_signal = token.startswith('<_')
                num = int(token.strip('<_>'))
                source_dict = signal_sources if is_signal else cycle_sources
                edge_type = 'signal' if is_signal else 'process'
                if num in source_dict:
                    G.add_edge(source_dict[num], last_node, type=edge_type)
    return G

# --- NETWORKX to SFILES ---
def networkx_to_compact_sfile(G):
    """Converts a NetworkX graph to the compact single-line format using DFS."""
    if not G: return ""
    starts = [n for n, d in G.in_degree() if d == 0]
    if not starts and list(G.nodes): starts = [list(G.nodes())[0]]
    
    visited, output_parts, node_list = set(), [], list(G.nodes)
    cycle_edges = set()
    
    def get_node_type_for_compact(node_id):
        match = re.match(r"([a-zA-Z_]+)-(\d+)", node_id)
        return match.groups()[0] if match else node_id

    def dfs_traversal(node):
        if node in visited:
            try: return f"<{node_list.index(node)+1}>" # Ring closing tag
            except ValueError: return ""
        visited.add(node)
        
        part = f"({get_node_type_for_compact(node)})"
        
        # Add opening cycle tags for any outgoing back-edges
        for u, v in G.out_edges(node):
            if v in visited and (u, v) not in cycle_edges:
                cycle_edges.add((u,v))
                try: part += f"{node_list.index(v)+1}" # Ring opening tag
                except ValueError: pass

        children = [v for v in G.successors(node) if (node, v) not in cycle_edges]
        
        if len(children) == 1:
            part += dfs_traversal(children[0])
        elif len(children) > 1:
            branches = [dfs_traversal(child) for child in children]
            part += "".join([f"[{branch}]" for branch in branches if branch])
        return part

    for start_node in starts:
        if start_node not in visited:
            output_parts.append(dfs_traversal(start_node))
            
    return "n|".join(output_parts)


# --- GRAPH FILE PARSING & VALIDATION ---

def parse_graph_file(uploaded_file):
    file_name = uploaded_file.name; G = None
    if file_name.endswith('.graphml'): G = nx.read_graphml(uploaded_file)
    elif file_name.endswith('.json'): G = nx.json_graph.node_link_graph(json.load(uploaded_file))
    else: st.error("Unsupported file type.")
    if not G: return None
    relabel_mapping = {node: str(node) for node in G.nodes() if not isinstance(node, (str, int, float))}
    if relabel_mapping:
        G = nx.relabel_nodes(G, relabel_mapping, copy=True)
        st.info(f"Sanitized {len(relabel_mapping)} non-standard node IDs.")
    return enrich_graph_with_inferred_attributes(G)

def validate_and_correct_graph(G):
    """Validates and corrects node names, now aware of complex controller formats."""
    relabel_mapping, validation_log = {}, []
    valid_pattern = re.compile(r"^[a-zA-Z_]+-\d+(?:/[A-Z_]+)?$")
    for node in G.nodes():
        if not valid_pattern.match(str(node)):
            simple_name = str(node).split('/')[0]
            match = re.match(r"([a-zA-Z_]+)(\d+)", simple_name)
            new_name = f"{match.groups()[0]}-{match.groups()[1]}" if match else f"{simple_name}-1"
            relabel_mapping[node] = new_name
            validation_log.append(f"FIXED: Invalid node '{node}'  -->  '{new_name}'")
    if relabel_mapping: return nx.relabel_nodes(G, relabel_mapping), validation_log
    else: return G, ["All node names are valid."]

# --- VISUALIZATION & EXPORT ---

def visualize_graph_auto_layout(G):
    dot = graphviz.Digraph('P&ID'); dot.attr('node', shape='box', style='rounded,filled');
    dot.attr('edge', color='gray40'); dot.attr(rankdir='LR')
    for node, data in G.nodes(data=True):
        node_type = data.get('type', 'UNKNOWN').lower()
        color = COLOR_MAP.get(node_type, COLOR_MAP['DEFAULT'])
        dot.node(node, data.get('label', node), fillcolor=color)
    for source, target, data in G.edges(data=True):
        style = "dashed" if data.get('type') == 'signal' else "solid"
        dot.edge(source, target, style=style)
    return dot

def visualize_graph_positional_layout(G):
    pos, node_colors, has_pos_data = {}, [], True
    for node, data in G.nodes(data=True):
        if 'xmin' in data and 'ymin' in data:
            pos[node] = ((data.get('xmin',0)+data.get('xmax',0))/2, -((data.get('ymin',0)+data.get('ymax',0))/2))
            node_colors.append(COLOR_MAP.get(data.get('type','DEFAULT').lower(), COLOR_MAP['DEFAULT']))
        else: has_pos_data = False; break
    if not has_pos_data: return None
    fig, ax = plt.subplots(figsize=(12, 8)); nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', style='solid')
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=500)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_color='black')
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=l.replace('_',' ').title(), markerfacecolor=c, markersize=12) for l, c in COLOR_MAP.items()]
    ax.legend(handles=legend_handles, title="Component Types", loc='best'); ax.set_title("Positional Layout Visualization", fontsize=16); plt.axis('off')
    return fig

def convert_fig_to_bytes(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight"); buf.seek(0)
    return buf

# --- STREAMLIT APP UI ---

st.title("🛠️ Advanced SFILES P&ID Converter")
st.write("A utility to convert between the compact single-line SFILES format and standard graph formats, following the official SFILES rules.")

tab1, tab2 = st.tabs(["SFILES to Graph", "Graph File to SFILES"])

with tab1:
    st.header("SFILES to Graph Converter")
    st.info("Paste your compact single-line SFILES content below. The parser understands component numbering `(hex){1}`, branches `[]`, cycles `<1` and `1`, and control signals `(C){FC}_1` and `<_1`.")
    # Corrected default string that matches the P&ID diagram
    default_compact = "(raw)(C){FC}_1(v)<_1(hex){1}(C){TC}_2(sep)[(C){PC}_3][(C){LC}_4][(v)<_3(prod)](C){FC}_5<_4(v)<_5(sep)[(C){PC}_6][(C){LC}_7][(v)<_6(prod)](C){FC}_8<_7(v)<_8(prod)"
    text_content = st.text_area("Compact SFILES Content", default_compact, height=150)
    
    # FIX: Changed key to be unique
    if st.button("Generate Graph from SFILES", key="gen_graph_from_sfiles"):
        if text_content:
            try:
                G = compact_sfile_to_networkx(text_content)
                st.session_state['graph_from_text'] = G
            except Exception as e:
                st.error(f"Failed to parse text: {e}")
                st.session_state['graph_from_text'] = None
        else: st.warning("Please enter some text content.")

    if st.session_state.get('graph_from_text'):
        G = st.session_state.get('graph_from_text')
        if G is not None:
            st.success("Graph generated successfully!")
            st.subheader("Visualized P&ID")
            dot_graph = visualize_graph_auto_layout(G)
            st.graphviz_chart(dot_graph)
            st.subheader("Download Options")
            c1, c2, c3 = st.columns(3)
            with c1: st.download_button("⬇️ Download GraphML", "\n".join(nx.generate_graphml(G)), "generated.graphml", "application/xml")
            with c2: st.download_button("⬇️ Download JSON", json.dumps(nx.json_graph.node_link_data(G), indent=4), "generated.json", "application/json")
            with c3: st.download_button("⬇️ Download PDF", dot_graph.pipe(format='pdf'), "generated.pdf", "application/pdf")

with tab2:
    st.header("Graph File to SFILES Converter")
    uploaded_file = st.file_uploader("Upload GraphML or JSON file", type=['graphml', 'json'])
    if uploaded_file:
        try:
            G_original = parse_graph_file(uploaded_file)
            if G_original:
                st.success("Graph file loaded and attributes inferred!")
                st.subheader("Node Name Validation")
                G_corrected, validation_log = validate_and_correct_graph(G_original)
                st.expander("Show Validation Log").text("\n".join(validation_log))
                st.subheader("Visualizations")
                dot_graph = visualize_graph_auto_layout(G_corrected)
                st.graphviz_chart(dot_graph)
                fig_matplotlib = visualize_graph_positional_layout(G_corrected)
                if fig_matplotlib: st.pyplot(fig_matplotlib)
                st.subheader("Convert to Compact SFILES Format")
                sfile_output = networkx_to_compact_sfile(G_corrected)
                st.text_area("Generated Text:", sfile_output, height=250)
                st.subheader("Download Options")
                cs, cj, cp, cpng = st.columns(4)
                with cs: st.download_button("⬇️ Download SFILES", sfile_output, "converted.txt", "text/plain")
                with cj: st.download_button("⬇️ Download JSON", json.dumps(nx.json_graph.node_link_data(G_corrected), indent=4), "converted.json", "application/json")
                with cp: st.download_button("⬇️ Download PDF", dot_graph.pipe(format='pdf'), "converted.pdf", "application/pdf")
                with cpng:
                    if fig_matplotlib: st.download_button("⬇️ Download PNG", convert_fig_to_bytes(fig_matplotlib), "converted.png", "image/png")
        except Exception as e: st.error(f"An error occurred: {e}")

