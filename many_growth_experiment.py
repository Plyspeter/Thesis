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
        if iterations == 100:
            hedwig.critical_info(info_str)
        return iterations == 100

    individual_generator.set_parameters(growth_iterations=7)
    
    experiment_averages = []
    experiment_bests = []
    
        
    for i in range(25):
        hedwig.critical_info(f"Experiment 7 iterations number: {i}")
        (best, average) = evo.run_evolution(callback)
        experiment_bests.append(best)
        experiment_averages.append(average)

    baseline_averages = []
    baseline_bests = []
    
    individual_generator.set_parameters(growth_iterations=3)

    for i in range(25):
        hedwig.critical_info(f"Experiment 3 iterations number: {i}")
        best, average = evo.run_evolution(callback)
        baseline_bests.append(best)
        baseline_averages.append(average)

    hedwig.critical_info(f'''
        NO HIDDEN EXPERIMENT 15 iterations
        Experiment Bests = {experiment_bests}
        Experiment Averages = {experiment_averages}
        Max(Experiment Bests): {max(experiment_bests)}
        Max(Experiment Averages): {max(experiment_averages)}
        Average(Experiment Bests): {numpy.average(experiment_bests)}
        Average(Experiment averages): {numpy.average(experiment_averages)}
    ''')

    hedwig.critical_info(f'''
        NO HIDDEN EXPERIMENT 3 iterations
        Baseline Bests: {baseline_bests}
        Baseline Averages = {baseline_averages}
        Max(Baseline Bests): {max(baseline_bests)}
        Max(Baseline Averages): {max(baseline_averages)}
        Average(Baseline Bests): {numpy.average(baseline_bests)}
        Average(Baseline Averages): {numpy.average(baseline_averages)}
    ''')

    hedwig.critical_info(f'''
        T_Test on averages: {stats.ttest_ind(experiment_averages, baseline_averages)}
        T_Test on bests: {stats.ttest_ind(experiment_bests, baseline_bests)}
    ''')



if __name__ == '__main__':
    main()