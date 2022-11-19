from copy import deepcopy
import random

from evolution.individual import Individual
import hedwig
import pickle

from network.nca import NCA

class NCAIndividual(Individual):
    
    def __init__(self, graph_generator, network_generator, nca=None, **config) -> None:
        super().__init__(graph_generator, network_generator, nca, **config)
        
    def _grow_graph(self):
        assert self._graph_n_input is not None and self._graph_n_output is not None
        hedwig.debug(f"Growing graph for individual: {self._id}")
        self._graph = self._graph_generator.get_graph(self._graph_n_input, self._graph_n_output)
        for i in range(self._growth_iterations):
            hedwig.debug(f"Starting growth iteration: {i}")
            for id in self._graph.get_inputs() + self._graph.get_ids():
                neighbourhood = self._graph.get_neighbourhood(id)
                new_neighbourhood = self._NCA.forward(neighbourhood)
                self._graph.update_graph(id, new_neighbourhood)

            hedwig.debug(f"All neighbourhoods updated, doing nagini correction")
            self._graph.nagini_correction()
            hedwig.debug(f"Ending growth iteration: {i}")
        
        self._graph.prune_islands()
        
        hedwig.debug(f"Graph grown for individual: {self._id}")
    
    def get_random_layer(self) -> int:
        num_layers = self.get_num_of_layers()
        layers = []
        weights = []
        for layer_id in range(num_layers):
            layers.append(layer_id)
            weights.append(self.get_num_of_neurons(layer_id))
        return random.choices(layers, weights=weights, k=1)[0]            
    
    def get_num_of_layers(self) -> int:
        return self._NCA.get_num_of_layers()
    
    def get_num_of_neurons(self, layer_id) -> int:
        return self._NCA.get_num_of_neurons(layer_id)

    def get_num_of_weights(self, layer_id, neuron_id) -> int:
        return self._NCA.get_num_of_weights(layer_id, neuron_id)
    
    def add_to_weight(self, layer_id: int, neuron_id: int, weight_id: int, val: float) -> None:
        self._NCA.add_to_weight(layer_id, neuron_id, weight_id, val)

    def add_to_bias(self, layer_id: int, neuron_id: int, val: float) -> None:
        self._NCA.add_to_bias(layer_id, neuron_id, val)
    
    def copy_for_mutation(self) -> 'NCAIndividual':
        copy = NCAIndividual(self._graph_generator, self._network_generator, deepcopy(self._NCA))
        copy._growth_iterations = self._growth_iterations
        return copy
    
    def save_nca(self, path) -> None:
        self._NCA.save(path + ".nca")
        super()._save(path)
    
    @staticmethod
    def load_nca(path) -> 'Individual':
        gg, ng, conf = Individual._load(path)
        #conf["growth_iterations"] = 7
        #conf["nca_topology"] = {
        #    "input_size": 251, 
        #    "output_size": 171, 
        #    "hidden_layer_sizes": [255,255,255,255], 
        #    "acts":["tanh", "tanh", "tanh", "tanh", "sigmoid"]
        #}
        return NCAIndividual(gg, ng, NCA.load(path + ".nca"), **conf)

    @staticmethod
    def load_shared_nca(path, name):
        gg, ng, conf = Individual._load_shared(path)
        return NCAIndividual(gg, ng, NCA.load(path + name + ".nca"), **conf)

    
    
        
    