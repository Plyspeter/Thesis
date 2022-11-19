import unittest
from network.nca import NCA

class TestNCA(unittest.TestCase):
    def test_one_hidden(self):
        torchNet = NCA(1, 1, [10], ["linear", "linear"])
        
        for i in range(torchNet.get_num_of_layers()):
            for j in range(torchNet.get_num_of_neurons(i)):
                for k in range(torchNet.get_num_of_weights(i, j)):
                    torchNet.set_weight(i, j, k, 1)
                torchNet.set_bias(i, j, 0)

        x = torchNet.forward([1.])

        self.assertEqual(x, [10])

    def test_many_hidden(self):
        torchNet = NCA(1, 1, [10, 10, 10, 10, 10], ["linear", "linear", "linear", "linear", "linear", "linear"])
        
        for i in range(torchNet.get_num_of_layers()):
            for j in range(torchNet.get_num_of_neurons(i)):
                for k in range(torchNet.get_num_of_weights(i, j)):
                    torchNet.set_weight(i, j, k, 1)
                torchNet.set_bias(i, j, 0)

        x = torchNet.forward([1.])

        self.assertEqual(x, [100000])
    
    def test_add_to_weight_and_bias(self):
        torchNet = NCA(1, 1, [10], ["linear", "linear"])
        
        for i in range(torchNet.get_num_of_layers()):
            for j in range(torchNet.get_num_of_neurons(i)):
                for k in range(torchNet.get_num_of_weights(i, j)):
                    torchNet.set_weight(i, j, k, 0)
                torchNet.set_bias(i, j, 1)

        for i in range(torchNet.get_num_of_layers()):
            for j in range(torchNet.get_num_of_neurons(i)):
                for k in range(torchNet.get_num_of_weights(i, j)):
                    torchNet.add_to_weight(i, j, k, 1)
                torchNet.add_to_bias(i, j, -1)

        x = torchNet.forward([1.])

        self.assertEqual(x, [10])