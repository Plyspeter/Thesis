import hedwig
from graph.graph import Graph

class GraphGenerator:
    
    def __init__(self)-> None:
        self.__config = {}
        pass
    
    def set_parameters(self, **config) -> None:
        hedwig.debug("Setting parameters for GraphGenerator")

        for key, val in config.items():
            self.__config[key] = val

        hedwig.debug("Parameters set")
                
    def get_parameters(self) -> dict:
        return self.__config
        
    def get_graph(self, num_in : int, num_out : int) -> Graph:
        return Graph(num_in, num_out, self.__config)
        