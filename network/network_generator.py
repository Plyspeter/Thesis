import hedwig
from network.pretty_neat_network import PrettyNeatNetwork


class NetworkGenerator:
    
    def __init__(self)-> None:
        self.__config = {}
        pass
    
    def set_parameters(self, **config) -> None:
        hedwig.debug("Setting parameters for NetworkGenerator")

        for key, val in config.items():
            self.__config[key] = val

        hedwig.debug("Parameters set")
        
    def get_parameters(self) -> dict:
        return self.__config
        
    def get_network(self, graph: 'dict[int, list[int]]', inputs: 'list[int]', hiddens: 'list[int]', outputs: 'list[int]', acts: 'dict[int, int]', bias: 'dict[int, int]', weights: 'dict[(int, int), float]') -> PrettyNeatNetwork:
        network = PrettyNeatNetwork()
        network.build(graph, inputs, hiddens, outputs, acts, bias, weights, self.__config)
        return network
        