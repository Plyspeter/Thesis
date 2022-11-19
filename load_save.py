from evolution.dcnca_individual import DCNCAIndividual
from evolution.nca_individual import NCAIndividual
from graph.graph_generator import GraphGenerator
from network.network_generator import NetworkGenerator
from evolution.individual_generator import IndividualGenerator
from config import config_reader
from evolution.evolution import Evolution
import numpy
import hedwig
from network.pretty_neat_network import PrettyNeatNetwork

def main():
    hedwig.init("logging_config.json")
    
    evo_config, ind_config, graph_config, neat_config = config_reader.read_config("config.json")
    
    network_generator = NetworkGenerator()
    network_generator.set_parameters(**neat_config)
    graph_generator = GraphGenerator()
    graph_generator.set_parameters(**graph_config)
    individual_generator = IndividualGenerator(graph_generator, network_generator)
    individual_generator.set_parameters(**ind_config)

    
    individual = individual_generator.get_individual()

    individual.set_graph_n_input_output(2,2)

    individual.save_nca("nca")
    individual.get_output_network()
    individual.draw_graph(True, False)

    ind = DCNCAIndividual.load_nca("nca")
    
    ind.set_graph_n_input_output(2,2)
    ind.get_output_network()
    ind.draw_graph(True, False)

main()