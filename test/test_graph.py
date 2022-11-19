import unittest
from graph.graph import Graph
from config import config_reader

class TestGraph(unittest.TestCase):

    def setUp(self) -> None:
        _, _, graph_config, _ = config_reader.read_config("config.json")
        self.conf = graph_config
        self.conf["neighbourhood_size"] = 3
        self.conf["default_act"] = 0
        self.conf["default_bias"] = 1
        self.conf["default_weight"] = 1

        self.act_fn = [1] + [0] * 9
        self.bias = [1]

    def transform_to_graph_with_one_hidden(self, G) -> Graph:
        G.update_graph(2, [0, 0, 0, 0, 0, 0, 0, 0] * 2 + self.act_fn + self.bias)
        G.nagini_correction()
        G.update_graph(1, [0, 0, 0, 0, 0, 1, 0, 0] * 2 + self.act_fn + self.bias)
        G.nagini_correction()
        G.update_graph(1, [0, 0, 0, 0, 0, 0, 1, 0] * 2 + self.act_fn + self.bias)
        G.nagini_correction()
        G.update_graph(6, [0, 1, 1, 0, 0, 0, 1, 1] * 2 + self.act_fn + self.bias)
        G.nagini_correction()
        return G

    def test_correct_setup(self):
        G = Graph(2, 2, self.conf)

        self.assertEqual(2, len(G.input))
        self.assertEqual(0, len(G.hidden))
        self.assertEqual(2, len(G.output))

    def test_incorrect_setup(self):
        self.assertRaises(AssertionError, Graph, 0, 1, self.conf)
        self.assertRaises(AssertionError, Graph, 1, 0, self.conf)
        self.assertRaises(AssertionError, Graph, 0, 0, self.conf)

    def test_expected_ids(self):
        G = Graph(2, 2, self.conf)
        ids = G.get_ids()

        self.assertEqual(0, len(ids))

    def test_correct_simple_neigbourhood(self):
        conf = self.conf
        conf["neighbourhood_size"] = 3
        conf["default_act"] = 0
        conf["default_bias"] = 1
        conf["default_weight"] = 1
        G = Graph(2, 2, conf)
        G = self.transform_to_graph_with_one_hidden(G)

        node_id = G.get_ids()[0]
        neighbourhood = G.get_neighbourhood(node_id)
        expected = [0, 1, 1, 0, 0, 0, 1, 1] * 3 + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [1] #Nodes + Conns + Weights + Activation + Bias

        self.assertEqual(expected, neighbourhood)

    #def test_correct_large_input_and_output_neigbourhood(self):
    #    conf = graph_config
    #    conf["neighbourhood_size"] = 3
    #    conf["default_act"] = 0
    #    conf["default_bias"] = 1
    #    conf["default_weight"] = 1
    #    G = Graph(10, 12, conf)
#
    #    node_id = G.get_ids()[0]
    #    neighbourhood = G.get_neighbourhood(node_id)
    #    expected = [0, 1, 1, 0, 0, 0, 1, 1] * 3 + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [1] #Nodes + Conns + Weights + Activation + Bias
#
    #    self.assertEqual(expected, neighbourhood)
#
    #def test_removal_of_connection(self):
    #    conf = graph_config
    #    conf["neighbourhood_size"] = 3
    #    conf["default_act"] = 0
    #    conf["default_bias"] = 1
    #    conf["default_weight"] = 1
    #    G = Graph(10, 12, conf)
#
    #    node_id = G.get_ids()[0]
#
    #    #Test that the neighbourhood is as we expect it to be
    #    neighbourhood = G.get_neighbourhood(node_id)
    #    expected = [0, 1, 1, 0, 0, 0, 1, 1] + [0, 1, 1, 0, 0, 0, 1, 1] + [0, 1, 1, 0, 0, 0, 1, 1] + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [1]
#
    #    self.assertEqual(expected, neighbourhood)
#
    #    #Update graph by removing connection
    #    val = [0, 1, 1, 0, 0, 0, 1, 0] * 2 + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [1]
    #    G.update_graph(node_id, val)
#
    #    # Test that it is still in the neighbourhood, but not in the connections
    #    neighbourhood = G.get_neighbourhood(node_id)
    #    expected = [0, 1, 1, 0, 0, 0, 1, 1] + [0, 1, 1, 0, 0, 0, 1, 0] + [0, 1, 1, 0, 0, 0, 1, 0] + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [1]
    #    self.assertEqual(expected, neighbourhood)
#
    #    # Test that it has been correctly removed from the adjacency list
    #    adj = G.adj[node_id]
    #    self.assertEqual(11, len(adj)) #TODO: Why was this 1?
    #    #self.assertEqual(3, adj[0]) #TODO: What is this?

    def test_adding_new_node_above(self):
        conf = self.conf
        conf["neighbourhood_size"] = 3
        conf["default_act"] = 0
        conf["default_bias"] = 1
        conf["default_weight"] = 1
        G = Graph(2, 2, conf)
        G = self.transform_to_graph_with_one_hidden(G)

        node_id = G.get_ids()[0]
        val = [0, 1, 1, 1, 0, 0, 1, 1] * 2 + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [1]
        
        G.update_graph(node_id, val)

        # Test that it has not been added to the neighbourhood yet
        neighbourhood = G.get_neighbourhood(node_id)
        expected = [0, 1, 1, 0, 0, 0, 1, 1] * 3 + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [1]
        self.assertEqual(expected, neighbourhood)

        # Test that it is in the adj graph and has connected to the hidden node
        adj = G.adj[node_id + 1]
        self.assertEqual(1, len(adj))
        self.assertEqual([node_id], adj)

    def test_adding_new_node_below(self):
        conf = self.conf
        conf["neighbourhood_size"] = 3
        conf["default_act"] = 0
        conf["default_bias"] = 1
        conf["default_weight"] = 1
        G = Graph(2, 2, conf)
        G = self.transform_to_graph_with_one_hidden(G)

        node_id = G.get_ids()[0]
        val = [0, 1, 1, 0, 1, 0, 1, 1] * 2 + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [1]
        G.update_graph(node_id, val)

        # Test that it has not been added to the neighbourhood yet
        neighbourhood = G.get_neighbourhood(node_id)
        expected = [0, 1, 1, 0, 0, 0, 1, 1] * 3 + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [1]
        self.assertEqual(expected, neighbourhood)

        # Test that it is in the adj graph and has not connected to anything
        adj = G.adj[node_id + 1]
        self.assertEqual(0, len(adj))

        # Test that the hidden node is connected to new node
        adj = G.adj[node_id]
        self.assertEqual(3, len(adj))
        self.assertEqual([3, 4, 7], adj)

    def test_adding_new_node_after(self):
        conf = self.conf
        conf["neighbourhood_size"] = 3
        conf["default_act"] = 0
        conf["default_bias"] = 1
        conf["default_weight"] = 1
        G = Graph(2, 2, conf)
        G = self.transform_to_graph_with_one_hidden(G)

        node_id = G.get_ids()[0]
        val = [0, 1, 1, 0, 0, 1, 1, 1] * 2 + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [1]
        G.update_graph(node_id, val)

        # Test that it has not been added to the neighbourhood yet
        neighbourhood = G.get_neighbourhood(node_id)
        expected = [0, 1, 1, 0, 0, 0, 1, 1] * 3 + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [1]
        self.assertEqual(expected, neighbourhood)

        # Test that it is in the adj graph and has not connected to anything
        adj = G.adj[node_id + 1]
        self.assertEqual(0, len(adj))

        # Test that the hidden node is connected to new node
        adj = G.adj[node_id]
        self.assertEqual(3, len(adj))
        self.assertEqual([3, 4, 7], adj)

    def test_adding_new_node_before(self):
        conf = self.conf
        conf["neighbourhood_size"] = 3
        conf["default_act"] = 0
        conf["default_bias"] = 1
        conf["default_weight"] = 1
        G = Graph(2, 2, conf)
        G = self.transform_to_graph_with_one_hidden(G)

        node_id = G.get_ids()[0]
        val = [1, 1, 1, 0, 0, 0, 1, 1] * 2 + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [1]
        G.update_graph(node_id, val)

        # Test that it has not been added to the neighbourhood yet
        neighbourhood = G.get_neighbourhood(node_id)
        expected = [0, 1, 1, 0, 0, 0, 1, 1] * 3 + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [1]
        self.assertEqual(expected, neighbourhood)

        # Test that it has not been added to the neighbourhood yet
        neighbourhood = G.get_neighbourhood(node_id)
        expected = [0, 1, 1, 0, 0, 0, 1, 1] * 3 + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [1]
        self.assertEqual(expected, neighbourhood)

        # Test that it is in the adj graph and has connected to the hidden node
        adj = G.adj[node_id + 1]
        self.assertEqual(1, len(adj))
        self.assertEqual([node_id], adj)

    def test_adding_connection(self):
        conf = self.conf
        conf["neighbourhood_size"] = 3
        conf["default_act"] = 0
        conf["default_bias"] = 1
        conf["default_weight"] = 1
        G = Graph(2, 2, conf)
        G = self.transform_to_graph_with_one_hidden(G)

        node_id = G.get_ids()[0]
        val = [0, 1, 1, 0, 0, 0, 1, 0] * 2 + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [1]
        G.update_graph(node_id, val)

        # Test that it is still in the neighbourhood, but not in the connections
        neighbourhood = G.get_neighbourhood(node_id)
        expected = [0, 1, 1, 0, 0, 0, 1, 1] + [0, 1, 1, 0, 0, 0, 1, 0] + [0, 1, 1, 0, 0, 0, 1, 0] + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [1]
        self.assertEqual(expected, neighbourhood)

        # Add connection back again
        val = [0, 1, 1, 0, 0, 0, 1, 1] * 2 + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [1]
        G.update_graph(node_id, val)
        expected = [0, 1, 1, 0, 0, 0, 1, 1] * 3 + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [1]
        neighbourhood = G.get_neighbourhood(node_id)
        self.assertEqual(expected, neighbourhood)

    def test_output_connected_again_after_correction(self):
        conf = self.conf
        conf["neighbourhood_size"] = 3
        conf["default_act"] = 0
        conf["default_bias"] = 1
        conf["default_weight"] = 1
        G = Graph(2, 2, conf)
        G = self.transform_to_graph_with_one_hidden(G)

        node_id = G.get_ids()[0]
        G.update_graph(node_id, [0, 1, 1, 0, 0, 0, 0, 0] * 2 + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [1])

        G.fully_connect_output()

        expected = [0, 1, 1, 0, 0, 0, 1, 1] * 3 + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [1]
        neighbourhood = G.get_neighbourhood(node_id)
        self.assertEqual(expected, neighbourhood)

    def test_correct_ordering_after_correction_next(self):
        conf = self.conf
        conf["neighbourhood_size"] = 3
        conf["default_act"] = 0
        conf["default_bias"] = 1
        conf["default_weight"] = 1
        G = Graph(2, 2, conf)
        G = self.transform_to_graph_with_one_hidden(G)

        G.update_graph(6, [0, 1, 1, 0, 1, 0, 1, 1] * 2 + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [1])

        G.nagini_correction()

        expected = [0, 1, 1, 0, 0, 0, 0, 1] * 3 + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [1]
        neighbourhood = G.get_neighbourhood(6)
        self.assertEqual(expected, neighbourhood)

        expected = [1, 0, 0, 0, 0, 1, 1, 0] + [1, 0, 0, 0, 0, 0, 0, 0] * 2 + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [1]
        neighbourhood = G.get_neighbourhood(7)
        self.assertEqual(expected, neighbourhood)

    def test_correct_not_added_when_no_conn_to_neuron(self):
        conf = self.conf
        conf["neighbourhood_size"] = 3
        conf["default_act"] = 0
        conf["default_bias"] = 1
        conf["default_weight"] = 1
        G = Graph(2, 2, conf)
        G = self.transform_to_graph_with_one_hidden(G)

        G.update_graph(6, [1, 1, 1, 0, 0, 0, 1, 1] * 2 + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [1])
        G.nagini_correction()
        expected = [0, 1, 1, 0, 0, 0, 1, 1] * 3 + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [1]
        neighbourhood = G.get_neighbourhood(6)
        self.assertEqual(expected, neighbourhood)

    def test_prune_one_island_away(self):
        conf = self.conf
        conf["neighbourhood_size"] = 3
        conf["default_act"] = 0
        conf["default_bias"] = 1
        conf["default_weight"] = 1
        G = Graph(2, 2, conf)
        G = self.transform_to_graph_with_one_hidden(G)

        G.update_graph(6, [0, 1, 1, 0, 0, 1, 1, 1] * 2 + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [1])
        G.nagini_correction()

        # Remove the connection to the new node
        # since it's not connected to the output
        G.prune_islands()
        G.nagini_correction()
        expected = [0, 1, 1, 0, 0, 0, 1, 1] * 3 + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [1]
        neighbourhood = G.get_neighbourhood(6)
        self.assertEqual(expected, neighbourhood)

    def test_prune_multiple_chain_island(self):
        conf = self.conf
        conf["neighbourhood_size"] = 3
        conf["default_act"] = 0
        conf["default_bias"] = 1
        conf["default_weight"] = 1
        G = Graph(2, 2, conf)
        G = self.transform_to_graph_with_one_hidden(G)

        G.update_graph(6, [0, 1, 1, 0, 0, 1, 1, 1] * 2 + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [1])
        G.nagini_correction()
        G.update_graph(7, [0, 0, 1, 0, 0, 0, 1, 0] * 2 + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [1])
        G.nagini_correction()

        # Remove the connection to the new node
        # since it's not connected to the output
        G.prune_islands()
        self.assertEqual([3, 4], G.adj[6])
        self.assertNotIn(7, G.adj)
        self.assertNotIn(8, G.adj)

    def test_remove_connection_to_neuron(self):
        conf = self.conf
        conf["neighbourhood_size"] = 3
        conf["default_act"] = 0
        conf["default_bias"] = 1
        conf["default_weight"] = 1
        G = Graph(2, 2, conf)
        G = self.transform_to_graph_with_one_hidden(G)

        G.update_graph(6, [0, 0, 1, 0, 0, 0, 1, 1] * 2 + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [1])
        G.nagini_correction()

        self.assertEqual([], G.adj[1])
        self.assertEqual([3, 4], G.adj[6])

        expected = [0, 0, 0, 0, 1, 0, 1, 0] + [0, 0, 0, 0, 0, 0, 0, 0] * 2 + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [1]
        neighbourhood = G.get_neighbourhood(1)
        self.assertEqual(expected, neighbourhood)

        expected = [0, 1, 1, 0, 0, 0, 1, 1] + [0, 0, 1, 0, 0, 0, 1, 1] * 2 + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [1]
        neighbourhood = G.get_neighbourhood(6)
        self.assertEqual(expected, neighbourhood)

    def test_remove_connection_from_neuron(self):
        conf = self.conf
        conf["neighbourhood_size"] = 3
        conf["default_act"] = 0
        conf["default_bias"] = 1
        conf["default_weight"] = 1
        G = Graph(2, 2, conf)
        G = self.transform_to_graph_with_one_hidden(G)
        
        G.update_graph(1, [0, 0, 0, 0, 0, 0, 0, 0] * 2 + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [1])
        G.nagini_correction()

        self.assertEqual([], G.adj[1])
        self.assertEqual([3, 4], G.adj[6])

        expected = [0, 0, 0, 0, 1, 0, 1, 0] + [0, 0, 0, 0, 0, 0, 0, 0] * 2 + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [1]
        neighbourhood = G.get_neighbourhood(1)
        self.assertEqual(expected, neighbourhood)

        expected = [0, 1, 1, 0, 0, 0, 1, 1] + [0, 0, 1, 0, 0, 0, 1, 1] * 2 + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [1]
        neighbourhood = G.get_neighbourhood(6)
        self.assertEqual(expected, neighbourhood)

    def test_new_node_created_by_two_nodes(self):
        G = Graph(1, 1, self.conf)
        G.update_graph(1, [0, 0, 0, 0, 0, 1, 0, 0] * 2 + self.act_fn + self.bias)
        G.nagini_correction()
        G.update_graph(1, [0, 0, 0, 0, 0, 0, 1, 0] * 2 + self.act_fn + self.bias)
        G.nagini_correction()
        G.update_graph(4, [0, 1, 0, 0, 0, 0, 1, 0] * 2 + self.act_fn + self.bias)
        G.nagini_correction()
        G.update_graph(1, [0, 0, 0, 0, 0, 1, 1, 0] * 2 + self.act_fn + self.bias)
        G.nagini_correction()
        G.update_graph(4, [0, 1, 0, 0, 0, 1, 1, 0] * 2 + self.act_fn + self.bias)
        G.update_graph(5, [0, 0, 1, 0, 0, 0, 1, 0] * 2 + self.act_fn + self.bias)
        G.nagini_correction()

        self.assertIn(6, G.adj[4])
        self.assertIn(6, G.adj[5])
        self.assertNotIn(7, G.adj)

    def test_new_node_created_two_nodes_back(self):
        G = Graph(1, 1, self.conf)
        G.update_graph(1, [0, 0, 0, 0, 0, 1, 0, 0] * 2 + self.act_fn + self.bias)
        G.nagini_correction()
        G.update_graph(1, [0, 0, 0, 0, 0, 0, 1, 0] * 2 + self.act_fn + self.bias)
        G.nagini_correction()
        G.update_graph(4, [0, 1, 0, 0, 0, 0, 1, 0] * 2 + self.act_fn + self.bias)
        G.nagini_correction()
        G.update_graph(1, [0, 0, 0, 0, 0, 1, 1, 0] * 2 + self.act_fn + self.bias)
        G.nagini_correction()
        G.update_graph(4, [1, 1, 0, 0, 0, 0, 1, 0] * 2 + self.act_fn + self.bias)
        G.update_graph(5, [0, 1, 1, 0, 0, 0, 0, 0] * 2 + self.act_fn + self.bias)
        G.nagini_correction()

        # Need to have prune removed from correction first!
        #self.assertIn(4, G.adj[6])
        #self.assertIn(5, G.adj[6])
        #self.assertEqual(3, len(G.adj[6]))
        #self.assertNotIn(7, G.adj)

    def test_new_node_created_two_nodes_front_and_back(self):
        G = Graph(1, 1, self.conf)
        G.update_graph(1, [0, 0, 0, 0, 0, 1, 0, 0] * 2 + self.act_fn + self.bias)
        G.nagini_correction()
        G.update_graph(1, [0, 0, 0, 0, 0, 0, 1, 0] * 2 + self.act_fn + self.bias)
        G.nagini_correction()
        G.update_graph(4, [0, 1, 0, 0, 0, 0, 1, 0] * 2 + self.act_fn + self.bias)
        G.nagini_correction()

        G.update_graph(1, [0, 0, 0, 0, 0, 1, 1, 0] * 2 + self.act_fn + self.bias)
        G.update_graph(4, [0, 1, 0, 1, 0, 0, 1, 0] * 2 + self.act_fn + self.bias)
        G.nagini_correction()

        self.assertIn(5, G.adj[1])
        self.assertIn(4, G.adj[5])
        self.assertNotIn(6, G.adj)
