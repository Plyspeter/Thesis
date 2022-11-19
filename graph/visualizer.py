# https://www.geeksforgeeks.org/visualize-graphs-in-python/
import networkx as nx
import matplotlib.pyplot as plt
from graph.plotly_draw import PlotlyDraw
from network.activation_function_converter import ActivationFunctionConverter

class GraphVisualizer:
    def __init__(self) -> None:
        self.G = nx.DiGraph()
        self.AFC = ActivationFunctionConverter()
        self.color_map = []
        self.weights = {}

    def add_edge(self, node_id_A, node_id_B, weight) -> None:
        self.G.add_edge(node_id_A, node_id_B)
        self.weights[(node_id_A, node_id_B)] = weight

    def add_node(self, id, pos, act_fn, bias) -> None:
        self.G.add_node(id, pos=(pos[0], -pos[1]), act_fn=self.AFC.convert_id_to_name(act_fn), bias=bias)

    def setup_colors(self, input_size, hidden_size, output_size) -> 'list[str]':
        self.color_map = ['green'] * input_size + ['red'] * output_size + ['grey'] * hidden_size

    def visualize(self, show, save) -> None:
        draw = PlotlyDraw(self.G, self.weights)
        
        if show:
            draw.show()
        if save:
            draw.save()



    #def visualize(self) -> None:
        #assert self.color_map != [], 'Need to setup colors before visualizing'

        #options = {
        #    "font_size": 14,
        #    "node_size": 700,
        #    "node_color": self.color_map,
        #    "edgecolors": "black",
        #    "linewidths": 1,
        #    "width": 1,
        #    "with_labels": True
        #}

        #draw(self.G, self.weights)

        # Uncomment to draw with networkx
        #nx.draw(self.G, pos=nx.get_node_attributes(self.G,'pos'), **options)
        #nx.draw_networkx_edge_labels(G, self.pos, edge_labels=self.edge_labels)
        #plt.show()