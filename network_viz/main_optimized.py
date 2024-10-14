import argparse
import networkx as nx
import pandas as pd
import colorsys
import random
from pyvis.network import Network

def generate_distinct_colors(n):
    hue_partition = 1.0 / (n + 1)
    colors = []
    for i in range(n):
        hue = i * hue_partition
        saturation = 0.7 + random.random() * 0.3
        lightness = 0.4 + random.random() * 0.2
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append('#%02x%02x%02x' % tuple(int(x * 255) for x in rgb))
    return colors

def save_network(stw_df, otu_df, network_name, edge_limit, base_node_size=20, base_font_size=30, scaling_factor=1.0):
    G = nx.Graph()

    edge_data = sorted(zip(stw_df['source'], stw_df['target'], stw_df['weight']), 
                       key=lambda x: abs(x[2]), reverse=True)[:edge_limit]

    for src, dst, weight in edge_data:
        G.add_edge(src, dst, weight=weight)

    degree = dict(G.degree())
    unique_nodes = set(G.nodes())
    colors = generate_distinct_colors(len(unique_nodes))
    color_map = dict(zip(unique_nodes, colors))

    # Use Fruchterman-Reingold force-directed algorithm
    pos = nx.spring_layout(G, k=1.5, iterations=50)

    net = Network(height="1000px", width="100%", bgcolor="#ffffff", font_color="#000000")

    max_degree = max(degree.values())
    min_degree = min(degree.values())
    max_weight = max(abs(weight) for _, _, weight in edge_data)

    for node in G.nodes():
        node_name = otu_df.columns[node]
        size = base_node_size + (degree[node] - min_degree) / (max_degree - min_degree) * base_node_size * 2
        size *= scaling_factor
        x, y = pos[node]
        net.add_node(node_name, label=node_name, title=node_name, color=color_map[node], 
                     size=size, x=x*3000, y=y*3000, font={'size': base_font_size, 'face': 'arial'})

    for src, dst, weight in edge_data:
        src_name = otu_df.columns[src]
        dst_name = otu_df.columns[dst]
        # edge_color = "#5170AE" if weight > 0 else "#DC8858"
        edge_color = "#DC8858" if weight > 0 else "#5170AE"
        width = 1 + (abs(weight) / max_weight) * 10  # Scale edge width
        net.add_edge(src_name, dst_name, value=abs(weight), color=edge_color, width=width)

    net.set_options("""
    var options = {
        "nodes": {
            "font": {
                "size": """ + str(base_font_size) + """,
                "face": "arial"
            }
        },
        "edges": {
            "smooth": {
                "type": "continuous",
                "forceDirection": "none"
            }
        },
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 100,
                "springConstant": 0.08
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": {"iterations": 150}
        },
        "interaction": {
            "dragNodes": true,
            "zoomView": true,
            "dragView": true
        }
    }
    """)

    net.show(f"{network_name}.html", notebook=False)

def main():
    parser = argparse.ArgumentParser(description="Generate enhanced network visualization for academic paper")
    parser.add_argument("--stw", required=True, help="Path to the source-target-weight CSV file")
    parser.add_argument("--otu", required=True, help="Path to the OTU CSV file")
    parser.add_argument("--edges", type=int, required=True, help="Number of top edges to include")
    parser.add_argument("--output", default="network_output", help="Output file name (without extension)")
    parser.add_argument("--base_node_size", type=float, default=20, help="Base size for nodes")
    parser.add_argument("--base_font_size", type=float, default=30, help="Base font size")
    parser.add_argument("--scaling_factor", type=float, default=1.0, help="Scaling factor for all sizes")

    args = parser.parse_args()

    stw_df = pd.read_csv(args.stw)
    otu_df = pd.read_csv(args.otu)

    save_network(stw_df, otu_df, args.output, args.edges, args.base_node_size, args.base_font_size, args.scaling_factor)

if __name__ == "__main__":
    main()
# python main_optimized.py --stw="/projects/genomic-ml/da2343/ml_project_1/model_complexity/source_target/ioral_GGM_source_target.csv" --otu="/projects/genomic-ml/da2343/ml_project_1/data/ioral_data_power_transformed.csv" --edges=200 --output="ioral_GGM_network"