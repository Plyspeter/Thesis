import sys
import os
from evolution.individual import Individual

__RENDER = False
if len(sys.argv) > 1 and sys.argv[1] == "HPC":
    os.chdir(os.getcwd() + "/thesis")
    __RENDER = False

from graph.graph_generator import GraphGenerator
from network.network_generator import NetworkGenerator
from evolution.individual_generator import IndividualGenerator
from config import config_reader
from evolution.fitness import Fitness
from evolution.evolution import Evolution
import hedwig

def main():
    hedwig.init("logging_config.json")
    
    evo_config, ind_config, graph_config, neat_config = config_reader.read_config("config.json")
    
    network_generator = NetworkGenerator()
    network_generator.set_parameters(**neat_config)
    graph_generator = GraphGenerator()
    graph_generator.set_parameters(**graph_config)
    individual_generator = IndividualGenerator(graph_generator, network_generator)
    individual_generator.set_parameters(**ind_config)
    evo = Evolution(individual_generator)
    evo.set_parameters(**evo_config)

    def callback(iterations : int, iterations_wo_improvement : int, pop : 'list[Individual]'):
        formatted_scores = list(map(lambda ind: (float) ("%.2f"%ind.get_score()), pop))
        bestIn10str = ""
        if iterations % 100 == 0:
            score = Fitness.run(evo_config["fitness"]["func"], pop[0], evo_config["fitness"]["vars"])
            bestIn10str = f"Best in 10 generations test score:  {score}"
        
        total_connections = f"Total number of connections: {len(pop[0].get_all_connections())}"
        dead_connections = f"Number of dead connections: {len(pop[0].find_dead_connections())}"

        hedwig.critical_info(f'''
    -------------------------------GENERATION:  {i}-------------------------------
    {formatted_scores}
    ------------------------------------------------------------------------------
    pop best score: {pop[0].get_score()}
    ------------------------------------------------------------------------------
    {total_connections}
    {dead_connections}
    ------------------------------------------------------------------------------
    {pop[0].get_activations_str()}
    {bestIn10str}
    ------------------------------------------------------------------------------
    ''')
        return False

    evo.run_evolution(callback)

if __name__ == '__main__':
    main()