import matplotlib.pyplot as plt
from pgmpy.models import BayesianNetwork
import networkx as nx

def visualize_bn(model):
    # Convert to networkx graph
    G = nx.DiGraph()

    G.add_edges_from(model.edges())

    plt.figure(figsize=(10, 6))

    pos = nx.spring_layout(G, seed=42)

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=3000,
        node_color="lightblue",
        font_size=10,
        arrowsize=20
    )

    plt.title("Bayesian Network Structure (Fraud Model)")
    plt.show()