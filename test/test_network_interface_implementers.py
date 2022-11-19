import unittest
from network.pretty_neat_network import PrettyNeatNetwork
import numpy as np

class TestPrettyNeat(unittest.TestCase):
    
    def test_big_fully_connected(self) -> None:
        i_size = 10
        self.h_size = 100
        o_size = 10
        i = list(np.arange(1, i_size+1, 1))
        o = list(np.arange(i_size+1, i_size + o_size + 1, 1))
        h = list(np.arange(i_size + o_size + 1, i_size + o_size + self.h_size +1, 1))
        
        i_to_h = {k:h for k in i}
        h_to_o = {k:o for k in h}
        adjacency_list = {**i_to_h, **h_to_o}
        
        weights = {(_to, _from):1 for _to, _from_list in adjacency_list.items() for _from in _from_list}
        bias = {b:0.5 for b in i + o + h}
        acts = {a:0 for a in i + o + h}

        neat_config = {"weight_multiplier": 1, "bias_multiplier": 1, "output_activation": 0}

        #Build all networks with linear act_fn, bias=0 and weights=1
        pretty_neat_network = PrettyNeatNetwork()
        pretty_neat_network.build(adjacency_list, i, h, o, acts, bias, weights, neat_config)
        
        inputs = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        actual = pretty_neat_network.run(inputs, o_size)
        expected = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
        self.assertEqual(expected, actual)
        
    def test_one_in_one_hidden_one_out(self):
        i = [1]
        h = [3]
        o = [2]
        
        adjacency_list = {1:[3], 3:[2]}
        weights = {(_to, _from):1 for _to, _from_list in adjacency_list.items() for _from in _from_list}
        bias = {b:0.5 for b in i + o + h}
        acts = {a:0 for a in i + o + h}

        neat_config = {"weight_multiplier": 1, "bias_multiplier": 1, "output_activation": 0}

        neat_network = PrettyNeatNetwork()
        neat_network.build(adjacency_list, i, h, o, acts, bias, weights, neat_config)
        
        expected = [2] 
        actual = neat_network.run([2], 1)
        self.assertEqual(expected, actual)
        
    def test_one_in_multiple_hidden_one_out(self):
        i = [1]
        h = [3, 4]
        o = [2]
        
        adjacency_list = {1:[3, 4], 3:[2], 4:[2]}
        weights = {(_to, _from):1 for _to, _from_list in adjacency_list.items() for _from in _from_list}
        bias = {b:0.5 for b in i + o + h}
        acts = {a:0 for a in i + o + h}

        neat_config = {"weight_multiplier": 1, "bias_multiplier": 1, "output_activation": 0}

        neat_network = PrettyNeatNetwork()
        neat_network.build(adjacency_list, i, h, o, acts, bias, weights, neat_config)
        
        expected = [2]
        actual = neat_network.run([1], 1)
        self.assertEqual(expected, actual)
        
    def test_one_in_multiple_hidden_multiple_out(self):
        i = [1]
        h = [4, 5, 6]
        o = [2, 3]
        
        adjacency_list = {1:[4, 5], 5:[2], 4:[2,3]}
        weights = {(_to, _from):1 for _to, _from_list in adjacency_list.items() for _from in _from_list}
        bias = {b:0.5 for b in i + o + h}
        acts = {a:0 for a in i + o + h}

        neat_config = {"weight_multiplier": 1, "bias_multiplier": 1, "output_activation": 0}

        neat_network = PrettyNeatNetwork()
        neat_network.build(adjacency_list, i, h, o, acts, bias, weights, neat_config)
        
        expected = [4,2]
        actual = neat_network.run([2], 2)
        self.assertEqual(expected, actual)
        
    def test_non_layered(self):
        i = [1]
        h = [4, 5, 6, 7]
        o = [2, 3]
        
        adjacency_list = {7:[3], 1:[4, 5, 7], 4:[6, 2, 3], 5:[6], 6:[2,3]}
        
        weights = {(_to, _from):1 for _to, _from_list in adjacency_list.items() for _from in _from_list}
        bias = {b:0.5 for b in i + o + h}
        acts = {a:0 for a in i + o + h}

        neat_config = {"weight_multiplier": 1, "bias_multiplier": 1, "output_activation": 0}

        neat_network = PrettyNeatNetwork()
        neat_network.build(adjacency_list, i, h, o, acts, bias, weights, neat_config)
        
        expected = [6,8]
        actual = neat_network.run([2], 2)
        self.assertEqual(expected, actual)

    def test_non_layered_complex(self):
        i = [1, 2]
        h = [5, 6, 7, 8, 9, 10, 11, 12]
        o = [3, 4]
        
        adjacency_list = {1:[5,6], 2:[7,8], 5:[10], 6:[9,10,11], 7:[9], 8:[9], 9:[12], 10:[4], 11:[12], 12:[3, 4]}
        
        weights = {(_to, _from):1 for _to, _from_list in adjacency_list.items() for _from in _from_list}
        bias = {b:0.5 for b in i + o + h}
        acts = {a:0 for a in i + o + h}

        neat_config = {"weight_multiplier": 1, "bias_multiplier": 1, "output_activation": 0}

        neat_network = PrettyNeatNetwork()
        neat_network.build(adjacency_list, i, h, o, acts, bias, weights, neat_config)     
        
        expected = [4,6]
        actual = neat_network.run([1, 1], 2)
        self.assertEqual(expected, actual)
    
    def test_non_layered_complex_9_to_11(self):
        i = [1, 2]
        h = [5, 6, 7, 8, 9, 10, 11, 12]
        o = [3, 4]
        
        adjacency_list = {1:[5,6], 2:[7,8], 5:[10], 6:[9,10,11], 7:[9], 8:[9], 9:[12, 11], 10:[4], 11:[12], 12:[3, 4]}
        
        weights = {(_to, _from):1 for _to, _from_list in adjacency_list.items() for _from in _from_list}
        bias = {b:0.5 for b in i + o + h}
        acts = {a:0 for a in i + o + h}

        neat_config = {"weight_multiplier": 1, "bias_multiplier": 1, "output_activation": 0}

        neat_network = PrettyNeatNetwork()
        neat_network.build(adjacency_list, i, h, o, acts, bias, weights, neat_config)  
        
        expected = [7,9]
        actual = neat_network.run([1, 1], 2)
        self.assertEqual(expected, actual)

    def test_non_layered_complex_11_to_9(self):
        i = [1, 2]
        h = [5, 6, 7, 8, 9, 10, 11, 12]
        o = [3, 4]
        
        adjacency_list = {1:[5,6], 2:[7,8], 5:[10], 6:[9,10,11], 7:[9], 8:[9], 9:[12], 10:[4], 11:[12, 9], 12:[3, 4]}
        weights = {(_to, _from):1 for _to, _from_list in adjacency_list.items() for _from in _from_list}
        bias = {b:0.5 for b in i + o + h}
        acts = {a:0 for a in i + o + h}

        neat_config = {"weight_multiplier": 1, "bias_multiplier": 1, "output_activation": 0}

        neat_network = PrettyNeatNetwork()
        neat_network.build(adjacency_list, i, h, o, acts, bias, weights, neat_config)  
        
        expected = [5,7]
        actual = neat_network.run([1, 1], 2)
        self.assertEqual(expected, actual)

    def test_non_layered_weight_2(self):
        i = [1]
        h = [4, 5, 6, 7]
        o = [2, 3]
        
        adjacency_list = {7:[3], 1:[4, 5, 7], 4:[6, 2, 3], 5:[6], 6:[2,3]}
        
        weights = {(_to, _from):1 for _to, _from_list in adjacency_list.items() for _from in _from_list}
        bias = {b:0.5 for b in i + o + h}
        acts = {a:0 for a in i + o + h}

        neat_config = {"weight_multiplier": 2, "bias_multiplier": 1, "output_activation": 0}

        neat_network = PrettyNeatNetwork()
        neat_network.build(adjacency_list, i, h, o, acts, bias, weights, neat_config)
        
        expected = [24, 32]
        actual = neat_network.run([2], 2)
        self.assertEqual(expected, actual)
    
    def test_non_layered_bias_1(self):
        i = [1]
        h = [4, 5, 6, 7]
        o = [2, 3]
        
        adjacency_list = {7:[3], 1:[4, 5, 7], 4:[6, 2, 3], 5:[6], 6:[2,3]}
        
        weights = {(_to, _from):1 for _to, _from_list in adjacency_list.items() for _from in _from_list}
        bias = {b:1 for b in i + o + h}
        acts = {a:0 for a in i + o + h}

        neat_config = {"weight_multiplier": 1, "bias_multiplier": 1, "output_activation": 0}

        neat_network = PrettyNeatNetwork()
        neat_network.build(adjacency_list, i, h, o, acts, bias, weights, neat_config)
        
        expected = [11, 14]
        actual = neat_network.run([2], 2)
        self.assertEqual(expected, actual)

    def test_non_layered_single_weight_2(self):
        i = [1]
        h = [4, 5, 6, 7]
        o = [2, 3]
        
        adjacency_list = {7:[3], 1:[4, 5, 7], 4:[6, 2, 3], 5:[6], 6:[2,3]}
        
        weights = {(_to, _from):0.75 for _to, _from_list in adjacency_list.items() for _from in _from_list}
        
        weights[1,4] = 1
        
        bias = {b:0.5 for b in i + o + h}
        acts = {a:0 for a in i + o + h}

        neat_config = {"weight_multiplier": 2, "bias_multiplier": 1, "output_activation": 0}

        neat_network = PrettyNeatNetwork()
        neat_network.build(adjacency_list, i, h, o, acts, bias, weights, neat_config)
        
        expected = [10, 12]
        actual = neat_network.run([2], 2)
        self.assertEqual(expected, actual)

    def test_non_layered_single_bias_1(self):
        i = [1]
        h = [4, 5, 6, 7]
        o = [2, 3]
        
        adjacency_list = {7:[3], 1:[4, 5, 7], 4:[6, 2, 3], 5:[6], 6:[2,3]}
        
        weights = {(_to, _from):1 for _to, _from_list in adjacency_list.items() for _from in _from_list}
        
        bias = {b:0.5 for b in i + o + h}
        bias[4] = 1

        acts = {a:0 for a in i + o + h}

        neat_config = {"weight_multiplier": 1, "bias_multiplier": 1, "output_activation": 0}

        neat_network = PrettyNeatNetwork()
        neat_network.build(adjacency_list, i, h, o, acts, bias, weights, neat_config)
        
        expected = [8, 10]
        actual = neat_network.run([2], 2)
        self.assertEqual(expected, actual)

    def test_non_layered_act_tanh(self):
        i = [1]
        h = [4, 5, 6, 7]
        o = [2, 3]
        
        adjacency_list = {7:[3], 1:[4, 5, 7], 4:[6, 2, 3], 5:[6], 6:[2,3]}
        
        weights = {(_to, _from):1 for _to, _from_list in adjacency_list.items() for _from in _from_list}
        
        bias = {b:0.5 for b in i + o + h}

        acts = {a:4 for a in i + o + h}

        neat_config = {"weight_multiplier": 1, "bias_multiplier": 1, "output_activation": 4}

        neat_network = PrettyNeatNetwork()
        neat_network.build(adjacency_list, i, h, o, acts, bias, weights, neat_config)
        
        expected = [np.tanh(np.tanh(np.tanh(2) + np.tanh(2)) + np.tanh(2)), np.tanh(np.tanh(np.tanh(2) + np.tanh(2)) + np.tanh(2) + np.tanh(2))]
        actual = neat_network.run([2], 2)
        self.assertEqual(expected, actual)

    def test_non_layered_single_act_tanh(self):
        i = [1]
        h = [4, 5, 6, 7]
        o = [2, 3]
        
        adjacency_list = {7:[3], 1:[4, 5, 7], 4:[6, 2, 3], 5:[6], 6:[2,3]}
        
        weights = {(_to, _from):1 for _to, _from_list in adjacency_list.items() for _from in _from_list}
        
        bias = {b:0.5 for b in i + o + h}

        acts = {a:0 for a in i + o + h}

        acts[6] = 4
        
        neat_config = {"weight_multiplier": 1, "bias_multiplier": 1, "output_activation": 0}

        neat_network = PrettyNeatNetwork()
        neat_network.build(adjacency_list, i, h, o, acts, bias, weights, neat_config)
        
        expected = [2 + np.tanh(4), 2 + 2 + np.tanh(4)]
        actual = neat_network.run([2], 2)
        self.assertEqual(expected, actual)


        