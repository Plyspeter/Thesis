from abc import ABC, abstractmethod
from graph import graph_generator
import hedwig
from network.nca import NCA
from network.network_generator import NetworkGenerator
from network.network_interface import NetworkInterface
from network.pretty_neat_network import PrettyNeatNetwork
import pickle

class Individual(ABC):
    
    ID = 0
    
    def __init__(self, graph_generator, network_generator, nca=None, **config) -> None:
        for key, val in config.items():
            setattr(self, "_" + key, val)

        self.__config = config
            
        self._id = Individual.ID
        Individual.ID += 1
        if nca is None:
            self._NCA = NCA(self._nca_topology["input_size"], self._nca_topology["output_size"], self._nca_topology["hidden_layer_sizes"], self._nca_topology["acts"])
        else:
            self._NCA = nca
        self._graph = None
        self._graph_n_input = None
        self._graph_n_output = None
        self._output_network = None
        self._score = None
        self._clean_score = None
        self._graph_generator = graph_generator
        self._network_generator = network_generator
        
    def get_id(self) -> int:
        return self._id
    
    @abstractmethod
    def _grow_graph(self):
        pass
        
    def get_config(self):
        return self.__config
        
    def _graph_to_network(self):
        if self._graph is None:
            self._grow_graph()
        hedwig.debug("Converting graph")
        adj, input, hidden, output, acts, bias, weights = self._graph.get_graph()
        self._output_network = self._network_generator.get_network(adj, input, hidden, output, acts, bias, weights)
        hedwig.debug("Graph converted")
        
    def find_dead_connections(self):
        assert self._graph is not None
        return self._graph.find_dead_connections()
    
    def get_all_connections(self):
        assert self._graph is not None
        return self._graph.get_all_connections()
    
    def get_n_unconnected(self) -> int:
        return self._graph.get_num_unconnected_nodes(self)
    
    def get_activations_str(self):
        assert self._graph is not None
        return self._graph.get_all_connections()
    
    def get_num_unconnected_nodes(self) -> int:
        return self._graph.get_num_unconnected_nodes()
    
    def any_hidden_nodes(self) -> bool:
        return self._graph.any_hidden_nodes()
        
    def get_output_network(self) -> PrettyNeatNetwork:
        if self._output_network is None:
            self._graph_to_network()
        return self._output_network
    
    @abstractmethod
    def get_random_layer(self) -> int:
        pass
    
    @abstractmethod
    def get_num_of_layers(self) -> int:
        pass
    
    @abstractmethod
    def get_num_of_neurons(self, layer_id) -> int:
        pass

    @abstractmethod
    def get_num_of_weights(self, layer_id, neuron_id) -> int:
        pass

    @abstractmethod
    def add_to_weight(self, layer_id: int, neuron_id: int, weight_id: int, val: float) -> None:
        pass
    
    @abstractmethod
    def add_to_bias(self, layer_id: int, neuron_id: int, val: float) -> None:
        pass
    
    def set_graph_n_input_output(self, n_input : int, n_output : int) -> None:
        self._graph_n_input = n_input
        self._graph_n_output = n_output
        
    def set_score(self, clean_score : int, penalty_score=0) -> None:
        self._score = clean_score - penalty_score
        self._clean_score = clean_score
        
    def get_score(self) -> int:
        assert self._score is not None
        return self._score
    
    def get_clean_score(self) -> int:
        assert self._clean_score is not None
        return self._clean_score
    
    def draw_graph(self, show : bool, save : bool) -> None:
        assert self._graph is not None
        
        self._graph.draw(show, save)
    
    @abstractmethod    
    def save_nca(self) -> None:
        pass
    
    def _save(self, path) -> None:
        pickle_out = open(path + ".gg", "wb")
        pickle.dump(self._graph_generator, pickle_out)
        pickle_out.close()
        
        pickle_out = open(path + ".ng", "wb")
        pickle.dump(self._network_generator, pickle_out)
        pickle_out.close()
        
        pickle_out = open(path + ".conf", "wb")
        pickle.dump(self.__config, pickle_out)
        pickle_out.close()
        
    @staticmethod
    def _load(path) -> 'tuple[graph_generator.GraphGenerator, NetworkGenerator, dict]':
        pickle_in = open(path + ".gg", "rb")
        gg = pickle.load(pickle_in)
        pickle_in.close()
        
        pickle_in = open(path + ".ng", "rb")
        ng = pickle.load(pickle_in)
        pickle_in.close()
        
        pickle_in = open(path + ".conf", "rb")
        conf = pickle.load(pickle_in)
        pickle_in.close()
        
        return (gg, ng, conf)
    
    @staticmethod
    def _load_shared(path):
        pickle_in = open(path + "gg", "rb")
        gg = pickle.load(pickle_in)
        pickle_in.close()
        
        pickle_in = open(path + "ng", "rb")
        ng = pickle.load(pickle_in)
        pickle_in.close()
        
        pickle_in = open(path + "conf", "rb")
        conf = pickle.load(pickle_in)
        pickle_in.close()
        
        return (gg, ng, conf)       

    @abstractmethod
    def copy_for_mutation(self) -> 'Individual':
        pass
    
        
    