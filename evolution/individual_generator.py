from evolution.dcnca_individual import DCNCAIndividual
from evolution.individual import Individual
from evolution.nca_individual import NCAIndividual
from evolution.nn_individual import NNIndividual
import hedwig

class IndividualGenerator:
    
    def __init__(self, graph_generator, network_generator)-> None:
        self.__config = {}
        self.__graph_generator = graph_generator
        self.__network_generator = network_generator
        pass
    
    def set_parameters(self, **config) -> None:
        hedwig.debug("Setting parameters for IndividualGenerator")

        for key, val in config.items():
            self.__config[key] = val

        hedwig.debug("Parameters set")
        
    def get_individual(self) -> Individual:
        if self.__config["kind"] == "normal":
            return NCAIndividual(self.__graph_generator, self.__network_generator, **self.__config)
        if self.__config["kind"] == "dc":
            return DCNCAIndividual(self.__graph_generator, self.__network_generator, **self.__config)
        if self.__config['kind'] == 'nn':
            return NNIndividual(self.__graph_generator, self.__network_generator, **self.__config)
        else:
            raise Exception("Individual kind not recognized")
        