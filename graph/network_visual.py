from network.network import Network
from graph.visualizer import GraphVisualizer

class NetworkVisualizer:
    def visualize(self, network : Network):
        lst = []
        lst.extend(network.get_input())
        lst.extend(network.get_hidden())

        G = GraphVisualizer()
        for neuron in lst:
            for (next, _) in neuron.get_next():
                G.add_edges(neuron, next)

        G.visualize()