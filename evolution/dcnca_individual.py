from abc import abstractmethod
from copy import deepcopy

import torch
from evolution.individual import Individual
import hedwig
from network.nca import NCA
import random

class DCNCAIndividual(Individual):
    
    def __init__(self, graph_generator, network_generator, nca=None, dc=None, **config) -> None:
        super().__init__(graph_generator, network_generator, nca, **config)
        if dc is None:
            self._DC = NCA(self._dc_topology["input_size"], self._dc_topology["output_size"], self._dc_topology["hidden_layer_sizes"], self._dc_topology["acts"])
        else:
            self._DC = dc
    
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
            
            hedwig.debug(f"Running Dead Connection NCA")
        
            dead_connections = self.find_dead_connections()
            for _from, _to in dead_connections:
                neighbourhood = self._graph.get_dead_connection_neighbourhood(_from, _to)
                new_neighbourhood = self._DC.forward(neighbourhood)
                self._graph.update_dead_connection(_from, _to, new_neighbourhood)
            hedwig.debug(f"Ending growth iteration: {i}")
        
        self._graph.prune_islands()
            
        hedwig.debug(f"Graph grown for individual: {self._id}")
        
    def __convert_layer_id(self, layer_id : int) -> 'tuple[NCA, int]':
        nca_layers = self._NCA.get_num_of_layers()
        if layer_id >= nca_layers:
            return self._DC, layer_id % nca_layers
        else:
            return self._NCA, layer_id
    
    def get_random_layer(self) -> int:
        num_layers = self.get_num_of_layers()
        layers = []
        weights = []
        for layer_id in range(num_layers):
            layers.append(layer_id)
            weights.append(self.get_num_of_neurons(layer_id))
        return random.choices(layers, weights=weights, k=1)[0]   
    
    def get_num_of_layers(self) -> int:
        nca_layers = self._NCA.get_num_of_layers()
        dc_layers = self._DC.get_num_of_layers()
        return nca_layers + dc_layers
    
    def get_num_of_neurons(self, layer_id) -> int:
        net, layer_id = self.__convert_layer_id(layer_id)
        return net.get_num_of_neurons(layer_id)

    def get_num_of_weights(self, layer_id, neuron_id) -> int:
        net, layer_id = self.__convert_layer_id(layer_id)
        return net.get_num_of_weights(layer_id, neuron_id)

    def add_to_weight(self, layer_id: int, neuron_id: int, weight_id: int, val: float) -> None:
        net, layer_id = self.__convert_layer_id(layer_id)
        net.add_to_weight(layer_id, neuron_id, weight_id, val)
    
    def add_to_bias(self, layer_id: int, neuron_id: int, val: float) -> None:
        net, layer_id = self.__convert_layer_id(layer_id)
        net.add_to_bias(layer_id, neuron_id, val)
    
    def copy_for_mutation(self) -> 'Individual':
        copy = DCNCAIndividual(self._graph_generator, self._network_generator, deepcopy(self._NCA), deepcopy(self._DC))
        copy._growth_iterations = self._growth_iterations
        return copy
    
    def save_nca(self, path) -> None:
        self._NCA.save(path + ".nca")
        self._NCA.save(path + ".dc")
        super()._save(path)
    
    @staticmethod
    def load_nca(path) -> 'Individual':
        gg, ng, conf = Individual._load(path)
        return DCNCAIndividual(gg, ng, NCA.load(path + ".nca"), NCA.load(path + ".dc"), **conf)
    
    @staticmethod
    def load_shared_nca(path, name):
        gg, ng, conf = Individual._load_shared(path)
        return DCNCAIndividual(gg, ng, NCA.load(path + name + ".nca"), NCA.load(path + name + ".dc"), **conf)
    