from evolution.fitness import Fitness
from evolution.individual_generator import IndividualGenerator
from graph.graph_generator import GraphGenerator
from gym_env.gym_env import GymEnv
import numpy as np
from config import config_reader

from network.network_generator import NetworkGenerator

# The random search algorithm builds n amount of NCA's and uses them to build
# n different graphs. These will be tested on the given gym enviornment and the
# best scoring will be returned.
class RandomSearch:

    def __init__(self):
        evo_config, ind_config, graph_config, neat_config = config_reader.read_config("config.json")
        self.pop_size = evo_config['pop_size']

        graph_gen = GraphGenerator()
        graph_gen.set_parameters(**graph_config)

        network_gen = NetworkGenerator()
        network_gen.set_parameters(**neat_config)

        self.ind_gen = IndividualGenerator(graph_gen, network_gen)
        self.ind_gen.set_parameters(**ind_config)

    def run(self, fitness_iterations, gym_env_name):
        scores = []
        env = GymEnv(gym_env_name)
        for _ in range(self.pop_size):
            ind = self.ind_gen.get_individual()
            config = {'fitness_iterations': fitness_iterations}
            scores.append(Fitness.fitness_nca(ind, env, **config))

        return (np.average(scores), np.max(scores))