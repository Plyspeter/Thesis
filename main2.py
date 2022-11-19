import os
import sys

__RENDER = False
if len(sys.argv) > 1 and sys.argv[1] == "HPC":
    os.chdir(os.getcwd() + "/thesis")
    __RENDER = False

from graph.graph_generator import GraphGenerator
from network.network_generator import NetworkGenerator
from evolution.individual_generator import IndividualGenerator
from config import config_reader
import numpy
from scipy import stats
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

    def callback(iterations, iterations_wo_improvements, pop):
        formatted_scores = list(map(lambda ind: (float) ("%.2f"%ind.get_score()), pop))
        bestIn10str = ""
        
        total_connections = f"Total number of connections: {len(pop[0].get_all_connections())}"
        dead_connections = f"Number of dead connections: {len(pop[0].find_dead_connections())}"
        
        info_str = f'''
-------------------------------GENERATION:  {iterations}-------------------------------
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
'''
        if iterations % 25 == 0:
            hedwig.critical_info(info_str)
        else:
            hedwig.info(info_str)
        return iterations == 100

    individual_generator.set_parameters(kind="dc")
    
    co_averages = []
    co_bests = []
    
        
    for i in range(25):
        hedwig.critical_info(f"Experiment number (Co): {i}")
        (best, average) = evo.run_evolution(callback)
        co_bests.append(best)
        co_averages.append(average)

    averages = []
    bests = []

    
    individual_generator.set_parameters(kind="normal")
    
    for i in range(25):
        hedwig.critical_info(f"Experiment number: {i}")
        best, average = evo.run_evolution(callback)
        bests.append(best)
        averages.append(average)

    hedwig.critical_info(f'''
        Co all bests = {co_bests}
        Co all averages = {co_averages}
        Co best score of all: {max(co_bests)}
        Co best average of all: {max(co_averages)}
        Co average best: {numpy.average(co_bests)}
        Co average average: {numpy.average(co_averages)}
    ''')

    hedwig.critical_info(f'''
        All bests: {bests}
        All averages = {averages}
        Best score of all: {max(bests)}
        Best average of all: {max(averages)}
        Average best: {numpy.average(bests)}
        Average average: {numpy.average(averages)}
    ''')

    hedwig.critical_info(f'''
        T_Test on averages: {stats.ttest_ind(co_averages, averages)}
        T_Test on bests: {stats.ttest_ind(co_bests, bests)}
    ''')



if __name__ == '__main__':
    main()