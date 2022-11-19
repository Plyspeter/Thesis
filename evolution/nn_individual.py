from evolution.individual import Individual
import random
from copy import deepcopy

from network.nca import NCA


class NNIndividual(Individual):

    def __init__(self, graph_generator, network_generator, nca=None, **config) -> None:
        super().__init__(graph_generator, network_generator, nca, **config)

    def _grow_graph(self):
        raise NotImplementedError

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
    
    def copy_for_mutation(self) -> 'NNIndividual':
        copy = NNIndividual(self._graph_generator, self._network_generator, deepcopy(self._NCA))
        copy._growth_iterations = self._growth_iterations
        return copy

    def get_output_network(self) -> NCA:
        return self._NCA

    def save_nca(self, path) -> None:
        raise NotImplementedError
    
    @staticmethod
    def load_nca(path) -> 'Individual':
        raise NotImplementedError