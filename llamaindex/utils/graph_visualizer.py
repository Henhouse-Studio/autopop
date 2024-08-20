import networkx as nx
import matplotlib.pyplot as plt
from llama_index.core import KnowledgeGraphIndex


def plot_index(index: KnowledgeGraphIndex):

    # Extract the graph data
    graph = index.get_networkx_graph()

    # Create a NetworkX graph
    G = nx.Graph()

    # Add nodes and edges to the graph
    for node, data in graph.nodes(data=True):
        G.add_node(node, label=data.get("label", node))

    for edge in graph.edges():
        G.add_edge(edge[0], edge[1])

    # Set up the plot
    plt.figure(figsize=(12, 8))

    # Draw the graph
    pos = nx.spring_layout(G)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=500,
        font_size=8,
        font_weight="bold",
    )

    # Add edge labels
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Show the plot
    plt.title("Knowledge Graph Visualization")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
