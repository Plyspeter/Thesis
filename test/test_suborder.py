import unittest
from graph.graph import Graph
from config import config_reader

class TestSubOrder(unittest.TestCase):

    def setUp(self) -> None:
        _, _, graph_config, _ = config_reader.read_config("config.json")
        self.conf = graph_config
        self.conf["neighbourhood_size"] = 3
        self.conf["default_act"] = 0
        self.conf["default_bias"] = 1
        self.conf["default_weight"] = 1
        self.conf["output_activation"] = 0
        
        self.act_fn = [1] + [0] * 9
        self.bias = [1]

    def transform_to_graph_with_one_hidden(self, G) -> Graph:
        G.update_graph(1, [0, 0, 0, 0, 0, 1, 0, 0] * 2 + self.act_fn + self.bias)
        G.nagini_correction()
        G.update_graph(1, [0, 0, 0, 0, 0, 0, 1, 0] * 2 + self.act_fn + self.bias)
        G.nagini_correction()
        G.update_graph(4, [0, 1, 0, 0, 0, 0, 1, 0] * 2 + self.act_fn + self.bias)
        G.nagini_correction()
        return G

    def test_probability(self):
        conf = self.conf
        conf["add_conn_threshold"]    = 0.8
        conf["remove_conn_threshold"] = 0.49
        G = Graph(1, 1, self.conf)
        G = self.transform_to_graph_with_one_hidden(G)
        val_input = [0, 0,   0,
                     0,      0,
                     0, 0.9, 0] * 2 + self.act_fn + self.bias

        val_hidd  = [0, 0.45, 0,
                     0,       0,
                     0, 1,    0] * 2 + self.act_fn + self.bias
        G.update_graph(1, val_input)
        G.update_graph(4, val_hidd)

        self.assertIn(4, G.adj[1])

    def test_insert_node_after(self):
        G = Graph(1, 1, self.conf)
        G = self.transform_to_graph_with_one_hidden(G)

        val = [0, 1, 0, 0, 0, 1, 1, 0] * 2 + self.act_fn + self.bias
        G.update_graph(4, val)
        G.nagini_correction()

        expected = [0, 1, 0, 0, 0, 1, 0, 0] * 3 + self.act_fn + self.bias
        neighbourhood = G.get_neighbourhood(4)
        self.assertEqual(expected, neighbourhood)

    def test_insert_node_after_lower(self):
        G = Graph(1, 1, self.conf)
        G = self.transform_to_graph_with_one_hidden(G)

        val = [0, 1, 0, 0, 0, 0, 1, 1] * 2 + self.act_fn + self.bias
        G.update_graph(4, val)
        G.nagini_correction()

        expected = [0, 1, 0, 0, 0, 0, 0, 1] * 3 + self.act_fn + self.bias
        neighbourhood = G.get_neighbourhood(4)
        self.assertEqual(expected, neighbourhood)

    def test_insert_node_before(self):
        G = Graph(1, 1, self.conf)
        G = self.transform_to_graph_with_one_hidden(G)

        val = [0, 1, 1, 0, 0, 0, 1, 0] * 2 + self.act_fn + self.bias
        G.update_graph(4, val)
        G.nagini_correction()

        # This currently does not work, since nodes cannot be inputed before.
        # This is WIP
        expected = [0, 1, 0, 0, 0, 0, 1, 0] * 3 + self.act_fn + self.bias
        neighbourhood = G.get_neighbourhood(4)
        print()
        print(expected)
        self.assertEqual(expected, neighbourhood)

    def test_node_pushed_forward(self):
        G = Graph(1, 1, self.conf)
        G = self.transform_to_graph_with_one_hidden(G)
        G.update_graph(1, [0, 0, 0, 0, 0, 1, 1, 0] * 2 + self.act_fn + self.bias)
        G.nagini_correction()
        G.update_graph(5, [0, 0, 1, 0, 1, 0, 0, 0] * 2 + self.act_fn + self.bias)
        G.nagini_correction()

        # Test right neigbourhoods
        expected = [0, 0, 0, 0, 0, 1, 0, 0] * 3 + self.act_fn + self.bias
        neighbourhood = G.get_neighbourhood(1)
        self.assertEqual(expected, neighbourhood)
        
        expected = [0, 1, 0, 0, 0, 0, 0, 0] * 3 + self.act_fn + self.bias
        neighbourhood = G.get_neighbourhood(2)
        self.assertEqual(expected, neighbourhood)

        expected = [1, 0, 0, 0, 0, 0, 1, 0] * 3 + self.act_fn + self.bias
        neighbourhood = G.get_neighbourhood(4)
        self.assertEqual(expected, neighbourhood)

        expected = [0, 0, 1, 0, 0, 0, 0, 1] * 3 + self.act_fn + self.bias
        neighbourhood = G.get_neighbourhood(5)
        self.assertEqual(expected, neighbourhood)

    def test_node_pushed_on_static(self):
        G = Graph(1, 1, self.conf)
        G = self.transform_to_graph_with_one_hidden(G)
        G.update_graph(1, [0, 0, 0, 0, 0, 1, 1, 0] * 2 + self.act_fn + self.bias)
        G.nagini_correction()
        G.update_graph(5, [0, 0, 1, 0, 1, 0, 0, 0] * 2 + self.act_fn + self.bias)
        G.nagini_correction()
        G.update_graph(1, [0, 0, 0, 0, 0, 1, 1, 0] * 2 + self.act_fn + self.bias)
        G.nagini_correction()
        G.update_graph(5, [0, 0, 1, 0, 1, 0, 0, 1] * 2 + self.act_fn + self.bias)
        G.nagini_correction()

        # Moving node has been moved down
        self.assertEqual((2, 1), G.lookup[6])
        # Static node has not been moved
        self.assertEqual((2, 0), G.lookup[4])

    def test_multiple_nodes_pushed_no_static(self):
        G = Graph(1, 1, self.conf)
        G = self.transform_to_graph_with_one_hidden(G)
        G.update_graph(1, [0, 0, 0, 0, 0, 1, 1, 0] * 2 + self.act_fn + self.bias)
        G.nagini_correction()
        G.update_graph(5, [0, 0, 1, 0, 1, 0, 0, 0] * 2 + self.act_fn + self.bias)
        G.nagini_correction()
        G.update_graph(1, [0, 0, 0, 0, 0, 1, 1, 0] * 2 + self.act_fn + self.bias)
        G.nagini_correction()
        G.update_graph(5, [0, 0, 1, 0, 1, 0, 0, 1] * 2 + self.act_fn + self.bias)
        G.update_graph(6, [0, 1, 0, 1, 0, 0, 1, 0] * 2 + self.act_fn + self.bias)
        G.nagini_correction()

        self.assertEqual((2, 0), G.lookup[6])
        self.assertEqual((3, 0), G.lookup[4])

    def test_multiple_static_nodes_hit(self):
        G = Graph(1, 1, self.conf)
        G = self.transform_to_graph_with_one_hidden(G)
        G.update_graph(1, [0, 0, 0, 0, 0, 1, 1, 0] * 2 + self.act_fn + self.bias)
        G.nagini_correction()
        G.update_graph(5, [0, 0, 1, 0, 1, 0, 0, 0] * 2 + self.act_fn + self.bias)
        G.nagini_correction()
        G.update_graph(1, [0, 0, 0, 0, 0, 1, 1, 0] * 2 + self.act_fn + self.bias)
        G.nagini_correction()
        G.update_graph(5, [0, 0, 1, 0, 1, 0, 0, 1] * 2 + self.act_fn + self.bias)
        G.nagini_correction()
        G.update_graph(1, [0, 0, 0, 0, 0, 1, 1, 0] * 2 + self.act_fn + self.bias)
        G.nagini_correction()
        G.update_graph(5, [0, 0, 1, 0, 1, 0, 0, 1] * 2 + self.act_fn + self.bias)
        G.nagini_correction()

        self.assertEqual((2, 0), G.lookup[4])
        self.assertEqual((2, 1), G.lookup[6])
        self.assertEqual((2, 2), G.lookup[7])
