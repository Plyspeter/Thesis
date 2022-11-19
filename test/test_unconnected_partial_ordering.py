import unittest
from graph.graph import Graph
from config import config_reader



class TestUnconnectedPartialOrdering(unittest.TestCase):

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

    def transform_to_graph_with_one_hidden(self, G) -> Graph:
        G.update_graph(1, [0, 0, 0, 0, 0, 1, 0, 0] * 2 + self.rest)
        G.nagini_correction()
        G.update_graph(1, [0, 0, 0, 0, 0, 0, 1, 0] * 2 + self.rest)
        G.nagini_correction()
        G.update_graph(4, [0, 1, 0, 0, 0, 0, 1, 0] * 2 + self.rest)
        G.nagini_correction()
        return G

    def test_one_backwards(self):
        G = Graph(1, 1, self.conf)
        G.update_graph(1, [0, 0, 0, 0, 0, 1, 1, 0] * 2 + self.rest)
        G.nagini_correction()
        G.update_graph(3, [0, 0, 1, 0, 0, 0, 1, 0] * 2 + self.rest)
        G.nagini_correction()
        G.update_graph(4, [1, 1, 0, 0, 0, 0, 0, 1] * 2 + self.rest)
        G.nagini_correction()

        self.assertEqual((1, -2), G.lookup[5])
        self.assertEqual(5, G.ordering[1][-2])

    def test_one_removed(self):
        G = Graph(1, 1, self.conf)
        G.update_graph(1, [0, 0, 0, 0, 0, 1, 1, 0] * 2 + self.rest) #new node over output + keep connection to output
        G.nagini_correction()
        G.update_graph(3, [0, 0, 1, 0, 0, 0, 1, 1] * 2 + self.rest) #keep connection to input + new node over output + connect to output
        G.nagini_correction()
        G.update_graph(4, [0, 1, 1, 0, 0, 0, 0, 1] * 2 + self.rest) #keep connection to 3 + new node over 3 + connect to output
        G.nagini_correction()
        G.update_graph(4, [0, 1, 0, 0, 0, 0, 0, 1] * 2 + self.rest) #keep connection to 3 + delete node 5 + keep connection to output
        G.nagini_correction()

        self.assertNotIn(5, G.adj) #5 has been deleted
        self.assertNotIn(5, G.lookup) #5 has been deleted
        self.assertNotIn(-2, G.ordering[1]) #nothing is above 3, because 5 has been deleted

    def test_forward_once(self):
        G = Graph(1, 1, self.conf)
        G.update_graph(1, [0, 0, 0, 0, 0, 1, 0, 0] * 2 + self.rest) # one above output + delete conn to output
        G.nagini_correction()
        G.update_graph(3, [0, 0, 1, 0, 0, 0, 1, 0] * 2 + self.rest) # keep conn to input + new node over output + delete conn to output
        G.nagini_correction()
        G.update_graph(4, [1, 1, 0, 0, 0, 0, 0, 1] * 2 + self.rest) # keep conn to 3 + new node over 3 + new conn to output
        G.nagini_correction()
        G.update_graph(5, [0, 0, 0, 0, 0, 0, 1, 1] * 2 + self.rest) # keep conn to 4 + new node over 4
        G.nagini_correction()

        self.assertEqual((2, -2), G.lookup[6])
        self.assertEqual(6, G.ordering[2][-2])
        self.assertIn(6, G.adj)

    def test_remove_on_input_order(self):
        G = Graph(1, 1, self.conf)
        G.update_graph(1, [0, 0, 0, 0, 0, 1, 0, 0] * 2 + self.rest)
        G.nagini_correction()
        G.update_graph(3, [1, 1, 0, 0, 0, 0, 0, 1] * 2 + self.rest)
        G.nagini_correction()

        self.assertNotIn(4, G.adj)
        self.assertNotIn(4, G.lookup)
        self.assertNotIn(-1, G.ordering[0])
        self.assertEqual(1, len(G.adj))
        self.assertEqual(0, len(G.adj[1]))

    def test_remove_connected_to_input_order_removal(self):
        G = Graph(1, 1, self.conf)
        G = self.transform_to_graph_with_one_hidden(G)
        G.update_graph(4, [0, 1, 0, 0, 0, 1, 1, 0] * 2 + self.rest)
        G.nagini_correction()
        G.update_graph(5, [1, 0, 1, 0, 0, 0, 0, 1] * 2 + self.rest)
        G.nagini_correction()
        G.update_graph(6, [0, 1, 0, 0, 0, 0, 0, 0] * 2 + self.rest)
        G.nagini_correction()

        self.assertNotIn(6, G.adj)
        self.assertNotIn(6, G.lookup)
        self.assertNotIn(-2, G.ordering[1])
        self.assertNotIn(7, G.adj)
        self.assertNotIn(7, G.lookup)
        self.assertNotIn(-2, G.ordering[0])

    def test_remove_more_then_one_size_island(self):
        G = Graph(1, 1, self.conf)
        G = self.transform_to_graph_with_one_hidden(G)
        G.update_graph(4, [0, 1, 0, 0, 0, 1, 1, 0] * 2 + self.rest)
        G.nagini_correction()
        G.update_graph(5, [0, 1, 1, 0, 0, 0, 0, 1] * 2 + self.rest)
        G.nagini_correction()
        G.update_graph(6, [0, 0, 0, 0, 0, 1, 0, 0] * 2 + self.rest)
        G.nagini_correction()

        self.assertNotIn(6, G.adj)
        self.assertNotIn(6, G.lookup)
        self.assertNotIn(-1, G.ordering[1])
        self.assertNotIn(7, G.adj)
        self.assertNotIn(7, G.lookup)
        self.assertNotIn(-2, G.ordering[2])

    def test_more_then_two_rec_forward(self):
        G = Graph(1, 1, self.conf)
        G = self.transform_to_graph_with_one_hidden(G)
        G.update_graph(4, [0, 1, 0, 0, 0, 1, 1, 0] * 2 + self.rest)
        G.nagini_correction()
        G.update_graph(5, [0, 1, 1, 0, 0, 0, 0, 1] * 2 + self.rest)
        G.nagini_correction()
        G.update_graph(6, [0, 0, 0, 0, 0, 1, 1, 0] * 2 + self.rest)
        G.nagini_correction()
        G.update_graph(7, [0, 0, 1, 0, 0, 0, 1, 0] * 2 + self.rest)
        G.nagini_correction()

        self.assertEqual((3, -2), G.lookup[8])

    def test_more_then_back_on_rec(self):
        G = Graph(1, 1, self.conf)
        G = self.transform_to_graph_with_one_hidden(G)
        G.update_graph(4, [0, 1, 0, 0, 0, 1, 1, 0] * 2 + self.rest)
        G.nagini_correction()
        G.update_graph(5, [0, 0, 1, 0, 0, 1, 0, 0] * 2 + self.rest)
        G.nagini_correction()
        G.update_graph(6, [0, 1, 1, 0, 0, 0, 0, 0] * 2 + self.rest)
        G.nagini_correction()
        G.update_graph(7, [0, 1, 0, 0, 0, 0, 1, 0] * 2 + self.rest)
        G.nagini_correction()

        self.assertEqual((1, -2), G.lookup[8])
        self.assertIn(8, G.adj)

    def test_more_then_back_on_rec_and_back_again(self):
        G = Graph(1, 1, self.conf)
        G = self.transform_to_graph_with_one_hidden(G)
        G.update_graph(4, [0, 1, 0, 0, 0, 1, 1, 0] * 2 + self.rest)
        G.nagini_correction()
        G.update_graph(5, [0, 0, 1, 0, 0, 1, 0, 0] * 2 + self.rest)
        G.nagini_correction()
        G.update_graph(6, [0, 1, 1, 0, 0, 0, 0, 0] * 2 + self.rest)
        G.nagini_correction()
        G.update_graph(7, [0, 1, 0, 0, 0, 0, 1, 0] * 2 + self.rest)
        G.nagini_correction()
        G.update_graph(8, [1, 0, 0, 0, 0, 1, 1, 0] * 2 + self.rest)
        G.nagini_correction()

        self.assertEqual((2, -3), G.lookup[10])
        self.assertIn(10, G.adj)
        self.assertNotIn(9, G.adj)
        self.assertNotIn(9, G.lookup)
        self.assertNotIn(-2, G.ordering[0])
    
    def test_going_down_on_you(self):
        G = Graph(1, 1, self.conf)
        G = self.transform_to_graph_with_one_hidden(G)
        G.update_graph(4, [0, 1, 0, 0, 0, 0, 1, 1] * 2 + self.rest)
        G.nagini_correction()
        G.update_graph(5, [1, 0, 1, 0, 0, 0, 0, 0] * 2 + self.rest)
        G.nagini_correction()

        self.assertEqual((1, 2), G.lookup[6])
        self.assertEqual(6, G.ordering[1][2])
        self.assertIn(6, G.adj)
        
    def test_network_with_backwards_subgraph(self):
        conf = self.conf
        conf["neighbourhood_size"] = 5
        conf["default_act"] = 0
        conf["default_bias"] = 1
        conf["default_weight"] = 1
        G = Graph(1, 1, self.conf)
        G.hidden.extend([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        G.adj = {1:[3], 3:[4], 4:[5], 5:[6], 6:[2], 7:[6, 8, 9], 10:[11, 7], 11:[12, 7], 12:[14], 9:[15], 13:[9, 15, 14], 14:[15], 15:[], 8:[]}
        G.lookup = {1:(0, 0), 3:(1, 0), 2:(2,0), 4:(None, 0), 5:(None, 0), 6:(None, 0), 7:(None, -1), 8:(None, -1), 10:(None, -1), 11:(None, -2), 12:(None, -2), 9:(None, -2), 15:(None, -2), 13:(None, -3), 14:(None, -3)}

        G.nagini_correction()

        self.assertEqual((3, -1), G.lookup[7])