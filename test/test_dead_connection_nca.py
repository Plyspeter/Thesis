import unittest
from graph.graph import Graph
from config import config_reader

class TestDeadConnectionNCA(unittest.TestCase):
    
    def setUp(self) -> None:
        _, _, graph_config, _ = config_reader.read_config("config.json")
        self.conf = graph_config
        self.conf["neighbourhood_size"] = 3
        self.conf["default_act"] = 0
        self.conf["default_bias"] = 1
        self.conf["default_weight"] = 1

        self.act_fn = [1] + [0] * 9
        self.bias = [1]
        self.rest = self.act_fn + self.bias

    
    def test_remove_1_dead_connection(self):
        G = Graph(1, 1, self.conf)
        G.update_graph(1, [0, 0, 0, 0, 0, 1, 1, 0] * 2 + self.rest)
        G.nagini_correction()
        G.update_graph(3, [0, 0, 1, 0, 0, 0, 1, 1] * 2 + self.rest)
        G.nagini_correction()
        G.update_dead_connection(1, 2, [0.0, 0.0])

        self.assertNotIn(2, G.adj[1])


    def test_change_1_dead_connection(self):
        G = Graph(1, 1, self.conf)
        G.update_graph(1, [0, 0, 0, 0, 0, 1, 1, 0] * 2 + self.rest)
        G.nagini_correction()
        G.update_graph(3, [0, 0, 1, 0, 0, 0, 1, 1] * 2 + self.rest)
        G.nagini_correction()
        G.update_dead_connection(1, 2, [1.0, 0.0])

        self.assertIn(2, G.adj[1])
        self.assertEqual(0.0, G.weights[1,2])