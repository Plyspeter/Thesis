import unittest
from graph.graph import Graph
from config import config_reader

class TestCenterOutputAroundInput(unittest.TestCase):

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

    def test_simple_1_1(self):
        G = Graph(1, 1, self.conf)
        self.assertEqual((0,0), G.lookup[1])
        self.assertEqual((1,0), G.lookup[2])

    def test_4_2(self):
        G = Graph(4, 2, self.conf)
        self.assertEqual((0,0), G.lookup[1])
        self.assertEqual((0,1), G.lookup[2])
        self.assertEqual((0,2), G.lookup[3])
        self.assertEqual((0,3), G.lookup[4])
        
        self.assertEqual((1,1), G.lookup[5])
        self.assertEqual((1,2), G.lookup[6])

    def test_2_4(self):
        G = Graph(2, 4, self.conf)
        self.assertEqual((0,0), G.lookup[1])
        self.assertEqual((0,1), G.lookup[2])
        
        self.assertEqual((1,-1), G.lookup[3])
        self.assertEqual((1,0), G.lookup[4])
        self.assertEqual((1,1), G.lookup[5])
        self.assertEqual((1,2), G.lookup[6])

    def test_3_1(self):
        G = Graph(3, 1, self.conf)
        self.assertEqual((0,0), G.lookup[1])
        self.assertEqual((0,1), G.lookup[2])
        self.assertEqual((0,2), G.lookup[3])
        
        self.assertEqual((1,1), G.lookup[4])
        

    def test_1_3(self):
        G = Graph(1, 3, self.conf)
        self.assertEqual((0,0), G.lookup[1])
        
        self.assertEqual((1,-1), G.lookup[2])
        self.assertEqual((1,0), G.lookup[3])
        self.assertEqual((1,1), G.lookup[4])

    def test_5_2(self):
        G = Graph(5, 2, self.conf)
        self.assertEqual((0,0), G.lookup[1])
        self.assertEqual((0,1), G.lookup[2])        
        self.assertEqual((0,2), G.lookup[3])
        self.assertEqual((0,3), G.lookup[4])
        self.assertEqual((0,4), G.lookup[5])
        
        self.assertEqual((1,1), G.lookup[6])
        self.assertEqual((1,2), G.lookup[7])

    def test_2_5(self):
        G = Graph(2, 5, self.conf)
        self.assertEqual((0,0), G.lookup[1])
        self.assertEqual((0,1), G.lookup[2])        
        
        self.assertEqual((1,-1), G.lookup[3])
        self.assertEqual((1,0), G.lookup[4])
        self.assertEqual((1,1), G.lookup[5])
        self.assertEqual((1,2), G.lookup[6])
        self.assertEqual((1,3), G.lookup[7])

        