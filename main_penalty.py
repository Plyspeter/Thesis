import os
import sys
import time
from evolution.fitness import Fitness

save = True
save_threshold = 499
__RENDER = False
if len(sys.argv) > 1 and sys.argv[1] == "HPC":
    os.chdir(os.getcwd() + "/thesis")
    __RENDER = False

from graph.graph_generator import GraphGenerator
from network.network_generator import NetworkGenerator
from evolution.individual_generator import IndividualGenerator
from config import config_reader
from evolution.evolution import Evolution
import numpy
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


    def callback(iterations : int, iterations_wo_improvement : int, pop):

        formatted_scores = list(map(lambda ind: (float) ("%.2f"%ind.get_score()), pop))

        bestIn10str = ""
        
        total_connections = f"Total number of connections: {len(pop[0].get_all_connections())}"
        dead_connections = f"Number of dead connections: {len(pop[0].find_dead_connections())}"
        
        info_str = f'''
-------------------------------KEYWORD----------------------------------------
-------------------------------GENERATION: {iterations}-------------------------------
{formatted_scores}
------------------------------------------------------------------------------
pop best clean score: {pop[0].get_clean_score()} with {pop[0].get_score()} dirty score
------------------------------------------------------------------------------
{total_connections}
{dead_connections}
------------------------------------------------------------------------------
{pop[0].get_activations_str()}
{bestIn10str}
------------------------------------------------------------------------------
'''     

        if iterations_wo_improvement == 0:
            hedwig.info(info_str)
        else:
            hedwig.info(info_str)
        if iterations_wo_improvement == 40:
            hedwig.critical_info(f'"No improvements in 40 generations. Ending experiment\n{info_str}')    
            callback.saved_ids.append(pop[0].get_id())
            print(callback.saved_ids)
            env_name = evo_config["fitness"]["vars"]["env_name"]
            pop[0].save_nca(f"AI/{env_name}_{pop[0].get_id()}_{pop[0].get_clean_score()}_{time.time()}")
            hedwig.critical_info(f'SCORE OF {pop[0].get_clean_score()} REACHED. SAVING NETWORK')
        return iterations_wo_improvement == 40

    callback.saved_ids = []

    averages = []
    bests = []
    env_name = evo_config["fitness"]["vars"]["env_name"]
    evo_config["fitness"]["vars"]["fitness_iterations"] = 100

    for n in range(25):
        hedwig.critical_info(f"{env_name} experiment number: {n}")
        pop = evo.run_evolution(callback)
        scores = []
        for individual in pop:
            scores.append(Fitness.run(evo_config["fitness"]["func"], individual, evo_config["fitness"]["vars"]))
        bests.append(max(scores))
        averages.append(numpy.average(scores))

    hedwig.critical_info(f'''
        {env_name} super test 10 bests: {bests}
        {env_name} super test 10 averages: {averages}
        -------------------------------------------------------------------------------------------------------------
        {env_name} super test 10 max best: {max(bests)}
        {env_name} super test 10 max average: {max(averages)}
        -------------------------------------------------------------------------------------------------------------
        {env_name} super test 10 average best: {numpy.average(bests)}
        {env_name} super test 10 average average: {numpy.average(averages)}
    ''')


if __name__ == '__main__':
    main()
