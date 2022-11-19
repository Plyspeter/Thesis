import sys
import os

if len(sys.argv) > 1 and sys.argv[1] == "HPC":
    os.chdir(os.getcwd() + "/thesis")

from graph.graph_generator import GraphGenerator
from network.network_generator import NetworkGenerator
from evolution.individual_generator import IndividualGenerator
from config import config_reader
from evolution.evolution import Evolution
from gym_env.supervised_env import SupervisedEnv
import hedwig

hedwig.init("logging_config.json")

hedwig.info("Starting supervised experiment!")
evo_config, ind_config, graph_config, neat_config = config_reader.read_config("config_supervised.json")
    
network_generator = NetworkGenerator()
network_generator.set_parameters(**neat_config)
graph_generator = GraphGenerator()
graph_generator.set_parameters(**graph_config)
individual_generator = IndividualGenerator(graph_generator, network_generator)
individual_generator.set_parameters(**ind_config)
evo = Evolution(individual_generator)
evo.set_parameters(**evo_config)

def callback(iterations : int, iterations_wo_improvements: int, population):
    best = population[0]
    score = best.get_score()
    env = SupervisedEnv(evo_config["fitness"]["vars"]["env_name"])
    acc = env.test_accuarcy(best.get_output_network())
    acc_val = env.test_validation_accuarcy(best.get_output_network())
    hedwig.info(f'Score: {score} -- Test Acc: {acc} -- Validation Acc: {acc_val}')

    if acc >= 99.99:
        hedwig.critical_info(f'Supervised learning found 100% accuarcy result!')
        return True

    if iterations_wo_improvements > 40:
        hedwig.info(f'No improvements in 40 generations, stopping test!')
        return True

    return False

evo.run_evolution(callback)