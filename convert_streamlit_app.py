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
    page_title="SFILES 2.0 P&ID Converter",
    page_icon="🛠️",
    layout="wide"
)

# --- SFILES 2.0 LOGIC & MAPPINGS (Based on Vogel et al. 2023) ---

COLOR_MAP = {
    'valve': '#FF6347', 'connector': '#4682B4', 'pipe': '#32CD32', 'mixer': '#FFD700',
    'pump': '#6A5ACD', 'splitter': '#FFA500', 'vessel': '#20B2AA', 'instrument': '#8A2BE2',
    'general': '#D3D3D3', 'tank': '#ADD8E6', 'heater': '#F08080', 'reactor': '#DA70D6',
    'raw_material': '#B0C4DE', 'product': '#98FB98', 'control': '#DDA0DD',
    'DEFAULT': '#808080', 'sep': '#A9A9A9', 'comp': '#A9A9A9', 'bout': '#A9A9A9', 'tout': '#A9A9A9'
}

PREFIX_MAP = {
    'v': ('VALVE', 'Valve'), 'pp': ('PUMP', 'Pump'), 'mix': ('MIXER', 'Mixer'),
    'splt': ('SPLITTER', 'Splitter'), 'r': ('REACTOR', 'Reactor'), 'tk': ('TANK', 'Tank'),
    'hex': ('HEATER', 'Heat Exchanger'), 'prod': ('PRODUCT', 'Product Stream'), 'raw': ('RAW_MATERIAL', 'Raw Material'),
    'c': ('CONTROL', 'Control'), 'sep': ('SEPARATOR', 'Separator'), 'comp': ('COMPRESSOR', 'Compressor'),
    'bout': ('OUTLET', 'Bottom Outlet'), 'tout': ('OUTLET', 'Top Outlet'), 
    'dist': ('DISTILLATION', 'Distillation System'), 'abs': ('ABSORPTION', 'Absorption Column')
}

CONTROLLER_TYPE_MAP = {
    'FC': 'Flow Controller', 'LC': 'Level Controller', 'PC': 'Pressure Controller',
    'TC': 'Temperature Controller', 'FI': 'Flow Indicator', 'PI': 'Pressure Indicator',
    'M': 'Manual Controller', 'LT': 'Level Transmitter'
}

def infer_attributes_from_id(node_id, controller_type=None):
    """
    Intelligently infers component TYPE and LABEL from its ID.
    """
    match = re.match(r"([a-zA-Z_]+)-(\d+)(?:/([A-Z_]+))?", str(node_id))
    
    prefix_only_match = re.match(r"([a-zA-Z_]+)", str(node_id))
    prefix = prefix_only_match.groups()[0].lower() if prefix_only_match else 'unknown'
    
    inferred_type, base_label = PREFIX_MAP.get(prefix, ('UNKNOWN', prefix.upper()))
    
    if match:
        prefix, number, subtype = match.groups()
        final_controller_type = controller_type if controller_type else subtype

        if prefix.lower() == 'c' and final_controller_type:
            inferred_type = 'CONTROL'
            base_label = CONTROLLER_TYPE_MAP.get(final_controller_type, f"Control {final_controller_type}")
        elif prefix.lower() in PREFIX_MAP:
            inferred_type, base_label = PREFIX_MAP.get(prefix.lower())
        
        return {'type': inferred_type, 'label': f"{base_label} {number}"}
        
    return {'type': inferred_type, 'label': str(node_id)}

def enrich_graph_with_inferred_attributes(G):
    # Ensure all nodes have non-None string attributes for GraphML compatibility
    for node, data in G.nodes(data=True):
        if 'type' not in data or 'label' not in data or data.get('type') is None:
            inferred_attrs = infer_attributes_from_id(node)
            data.setdefault('type', inferred_attrs['type'])
            data.setdefault('label', inferred_attrs['label'])
        
        for key, value in list(data.items()):
            if value is None:
                data[key] = ''
            elif not isinstance(value, (str, int, float, bool)):
                 data[key] = str(value)

    return G

# --- SFILES 2.0 PARSER (FIXED: Handles max() iterable error) ---

def compact_sfile_to_networkx(sfile_content):
    """
    Parses SFILES 2.0 string, ensuring unique IDs are generated correctly and stably.
    """
    G = nx.DiGraph()
    # Normalize separator: replace '|' with 'n|' if it's not already 'n|'
    normalized_content = sfile_content.replace('|', 'n|')
    if normalized_content.startswith('n'): normalized_content = normalized_content[1:]
    
    # Global counter initialization: Must track max used number to avoid accidental reuse
    global_max_counts = Counter() 
    
    for part in normalized_content.split('n|'):
        content = part.strip().replace('\n', '')
        if not content: continue
        
        token_pattern = re.compile(r'(\([a-zA-Z_]+\)(?:\{[a-zA-Z_0-9]+\})?(?:\{[a-zA-Z_0-9]+\})?|\[|\]|<_?[a-zA-Z_0-9]+>|_?[a-zA-Z_0-9]+|<&|&)')
        tokens = [match.group(0) for match in token_pattern.finditer(content)]
        
        path_stack, last_node, current_train_counts = [], None, Counter()
        cycle_sources, signal_sources, conv_sources = {}, {}, {}
        
        # This counter tracks component numbering within the current train/block.
        # It is reset for each n| block to avoid reusing implicit counter IDs.
        
        is_start_of_train = True 

        for token in tokens:
            if token == '[':
                if last_node:
                    path_stack.append(last_node)
                    
            elif token == ']':
                if path_stack:
                    path_stack.pop()

            elif token.startswith('('):
                comp_match = re.match(r'\(([a-zA-Z_]+)\)(?:\{([a-zA-Z_0-9]+)\})?(?:\{([a-zA-Z_0-9]+)\})?', token)
                comp_type, first_tag, second_tag = comp_match.groups() if comp_match else (None, None, None)
                
                num_str = second_tag if second_tag and second_tag.isdigit() else (first_tag if first_tag and first_tag.isdigit() else None)
                tag_str = None
                if first_tag and not first_tag.isdigit(): tag_str = first_tag
                elif second_tag and not second_tag.isdigit(): tag_str = second_tag

                controller_type = tag_str if comp_type.lower() == 'c' else None
                
                # --- Node ID Assignment Logic ---
                if num_str is None:
                    # Case: Implicit numbering. Use the global_max_counts to ensure a unique ID is created.
                    # FIX: Safely determine the max used number for this component type globally.
                    existing_numbers = [G.nodes[n].get('number') for n in G.nodes if n.startswith(comp_type + '-')]
                    max_existing = max([int(n) for n in existing_numbers if isinstance(n, (str, int)) and str(n).isdigit()] + [0])
                    
                    global_max_counts[comp_type] = max(global_max_counts[comp_type], max_existing) + 1
                    node_id = f"{comp_type}-{global_max_counts[comp_type]}"
                    node_number = global_max_counts[comp_type]
                else:
                    # Case: Explicit numbering. Use the number.
                    node_id = f"{comp_type}-{num_str}"
                    node_number = int(num_str)
                    # Update global counter to prevent implicit collision
                    if num_str.isdigit():
                        global_max_counts[comp_type] = max(global_max_counts[comp_type], int(num_str))


                # Add node if new (or enrich if existing)
                if not G.has_node(node_id):
                    attrs = infer_attributes_from_id(node_id, controller_type=controller_type)
                    G.add_node(node_id, **attrs)
                
                G.nodes[node_id]['number'] = node_number

                # Edge Connection Logic
                if last_node and last_node != node_id:
                    ancestor = path_stack[-1] if path_stack else None
                    
                    if ancestor == last_node:
                        G.add_edge(last_node, node_id, type='process', tag=tag_str if tag_str else '')
                    else:
                        G.add_edge(last_node, node_id, type='process', tag=tag_str if tag_str else '')
                        
                # Ensure edge tag is non-None
                for u, v, data in G.edges(data=True):
                    if data.get('tag') is None: data['tag'] = ''
                    if data.get('type') is None: data['type'] = 'process'
                
                last_node = node_id
                is_start_of_train = False

            # Cycle and Signal Tag Handling (Source Tags)
            elif re.match(r'^(?:_?[a-zA-Z0-9]+|\&)$', token): 
                num = token.strip('_')
                if not last_node: continue 
                
                if token.startswith('_'): signal_sources[num] = last_node
                elif token == '&': conv_sources['&'] = last_node
                else: cycle_sources[num] = last_node
                is_start_of_train = False

            # Target Tags
            elif re.match(r'^<.+>$', token):
                num = token.strip('<>')
                
                if num.startswith('_'): # Signal connection target <_#
                    num = num.strip('_')
                    if num in signal_sources and last_node: G.add_edge(signal_sources[num], last_node, type='signal', tag='')
                elif num == '&': # Converging branch target <&...&
                    conv_target = conv_sources.get('<&')
                    conv_source = conv_sources.get('&')
                    if conv_source and conv_target: 
                        G.add_edge(conv_source, conv_target, type='process', tag='')
                else: # Recycle target <#
                    if num in cycle_sources and last_node: G.add_edge(cycle_sources[num], last_node, type='process', tag='')
                is_start_of_train = False
        
    return enrich_graph_with_inferred_attributes(G)

# --- NETWORKX to SFILES ---
def networkx_to_compact_sfile(G):
    if not G: return ""
    starts = [n for n, d in G.in_degree() if d == 0]
    if not starts and list(G.nodes): starts = [list(G.nodes())[0]]
    
    visited, output_parts, node_list = set(), [], list(G.nodes)
    
    def get_node_type_for_compact(node_id):
        match = re.match(r"([a-zA-Z_]+)-(\d+)", node_id)
        if match:
            prefix = match.groups()[0]
            number = match.groups()[1]
            return f"({prefix}){{{number}}}"
        return f"({node_id.split('-')[0].lower()})"

    def dfs_traversal(node):
        if node in visited:
            try: return f"<{node_list.index(node)+1}>" 
            except ValueError: return ""
            
        visited.add(node)
        
        part = get_node_type_for_compact(node)
        
        cycle_tags = []
        children = []
        for u, v, data in G.out_edges(node, data=True):
            edge_type = data.get('type', 'process')
            stream_tag = data.get('tag', '')
            
            if v in visited:
                tag = f"_{node_list.index(v)+1}" if edge_type == 'signal' else str(node_list.index(v)+1)
                cycle_tags.append(tag)
            else:
                children.append((v, stream_tag))

        part += "".join(cycle_tags)
        
        if len(children) == 1:
            child, stream_tag = children[0]
            tag_prefix = f"{{{stream_tag}}}" if stream_tag else ""
            part += tag_prefix + dfs_traversal(child)
        elif len(children) > 1:
            children.sort(key=lambda x: G.nodes[x[0]].get('rank', x[0])) 
            
            branches = []
            for child, stream_tag in children[:-1]:
                tag_prefix = f"{{{stream_tag}}}" if stream_tag else ""
                branches.append(f"[{tag_prefix}{dfs_traversal(child)}]")
                
            last_child, last_stream_tag = children[-1]
            last_tag_prefix = f"{{{last_stream_tag}}}" if last_stream_tag else ""
            branches.append(last_tag_prefix + dfs_traversal(last_child))
            
            part += "".join(branches)
            
        return part

    for start_node in starts:
        if start_node not in visited:
            output_parts.append(dfs_traversal(start_node))
            
    return "n|".join([p for p in output_parts if p])


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
    relabel_mapping, validation_log = {}, []
    valid_pattern = re.compile(r"^[a-zA-Z_]+-\d+(?:/[A-Z_]+)?$")
    node_prefix_counts = Counter()
    for node, data in G.nodes(data=True):
        node_str = str(node)
        if not valid_pattern.match(node_str):
            label_parts = data.get('label', node_str).split()
            if len(label_parts) >= 2:
                prefix_map_lookup = {v[1].lower(): k for k, v in PREFIX_MAP.items()}
                prefix = prefix_map_lookup.get(label_parts[0].lower(), label_parts[0].lower())
                if label_parts[1].isdigit(): number = label_parts[1]
                else:
                    node_prefix_counts[prefix] += 1
                    number = node_prefix_counts[prefix]
                new_name = f"{prefix}-{number}"
                relabel_mapping[node] = new_name
                validation_log.append(f"FIXED: Invalid node '{node}' (Label: {data.get('label', 'N/A')}) --> '{new_name}'")
            else:
                node_prefix_counts['general'] += 1
                new_name = f"general-{node_prefix_counts['general']}"
                relabel_mapping[node] = new_name
                validation_log.append(f"FIXED: Highly non-standard node '{node}' --> '{new_name}'")
        else:
            match = re.match(r"([a-zA-Z_]+)-(\d+)", node_str)
            if match:
                prefix, number = match.groups()
                node_prefix_counts[prefix.lower()] = max(node_prefix_counts[prefix.lower()], int(number))

    if relabel_mapping: 
        G_new = nx.relabel_nodes(G, relabel_mapping, copy=True)
        return enrich_graph_with_inferred_attributes(G_new), validation_log
    else: 
        return G, ["All node names are valid."]


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
        label = data.get('tag', '')
        dot.edge(source, target, label=label, style=style)
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

st.title("🛠️ SFILES 2.0 P&ID Converter")
st.write("A utility to convert between the compact single-line SFILES 2.0 format and standard graph formats, following the rules proposed by Vogel et al. (2023).")

tab1, tab2 = st.tabs(["SFILES to Graph", "Graph File to SFILES"])

with tab1:
    st.header("SFILES to Graph Converter")
    st.info("The parser is configured to strictly follow SFILES 2.0 rules, meaning repeated numbered components in the string refer to the same node (reuse/convergence).")
    
    # Original problematic string (with non-standard tags and separator)
    default_complex_sfiles = "(raw)(hex){1}(C){TC}_1(hex){2}(mix)<2(r)<_2[(C){TC}_2][(C){LC}_3][{tout}(C){PC}_4(v)<_4(prod)]{bout}(v)<_3(splt)[(hex){2}(hex){3}(C){TC}_5(pp)[(C){M}](C){PI}(C){FC}_6(v)<_6(mix)<1(r)<_7[(C){TC}_7][(C){LC}_8][{bout}(v)<_8(prod)]{tout}(C){PC}_9(v)<_9(splt)[(C){FC}_10(v)1<_10](hex){4}(r)<_11[(C){TC}_11][(C){LC}_12][{tout}(C){PC}_13(v)<_13(prod)]{bout}(v)<_12(hex){4}(prod)](C){FC}_14(v)2<_14n|(raw)(hex){1}(v)<_1(prod)n|(raw)(hex){3}(v)<_5(prod)"

    text_content = st.text_area("Compact SFILES Content (Using standard numerical tags for best results)", default_complex_sfiles, height=150, key="sfiles_text_input")
    
    if st.button("Generate Graph from SFILES", key="gen_graph_from_sfiles"):
        text_content = st.session_state["sfiles_text_input"]
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
            st.success(f"Graph generated successfully! Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
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
                st.text_area("Generated SFILES:", sfile_output, height=250)
                
                st.subheader("Download Options")
                cs, cj, cp, cpng = st.columns(4)
                with cs: st.download_button("⬇️ Download SFILES", sfile_output, "converted.txt", "text/plain")
                with cj: st.download_button("⬇️ Download JSON", json.dumps(nx.json_graph.node_link_data(G_corrected), indent=4), "converted.json", "application/json")
                with cp: st.download_button("⬇️ Download PDF", dot_graph.pipe(format='pdf'), "converted.pdf", "application/pdf")
                with cpng:
                    if fig_matplotlib: st.download_button("⬇️ Download PNG", convert_fig_to_bytes(fig_matplotlib), "converted.png", "image/png")
        except Exception as e: st.error(f"An error occurred: {e}")
