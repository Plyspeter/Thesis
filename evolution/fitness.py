from copy import deepcopy
import time

import actual_multiprocessing as amp
from evolution.individual import Individual
from gym_env.gym_env import GymEnv
from gym_env.supervised_env import SupervisedEnv
import numpy as np
import hedwig

from inspect import getmembers, isfunction    

class Fitness:

    @staticmethod
    def run(method, individual, args):
        func = Fitness.get_function(method)
        env = None
        if args["env_kind"] == "gym":
            env = GymEnv(args["env_name"])
        elif args["env_kind"] == "supervised":
            env = SupervisedEnv(args["env_name"])
        assert env is not None
        return func(individual, env, **args)
    
    @staticmethod
    def get_function(name):
        for func_name, func in getmembers(Fitness, isfunction):
            if func_name == name:

                return func
    
    @staticmethod
    def fitness_nca(individual: Individual, env, **args):
        id = individual.get_id()
        hedwig.debug(f"Calculating fitness for individual: {id}")
        individual.set_graph_n_input_output(env.get_input_size(), env.get_output_size())
        output_network = individual.get_output_network()
        start = time.time()
        scores = amp.ActualPool(25, "fitness").run(env.run_network, [output_network] * args["fitness_iterations"])
        score = np.average(scores)
        end = time.time()
        #hedwig.info(f"individual {id:4} got fitness {score:8.2f} on network with {len(output_network.network.node[0]):4} nodes with time {end - start:8.3f}")
        individual.set_score(score)
        return score
    
    @staticmethod
    def fitness_penalize_trivial(individual: Individual, env, **args):
        id = individual.get_id()
        hedwig.debug(f"Calculating fitness for individual: {id}")
        individual.set_graph_n_input_output(env.get_input_size(), env.get_output_size())
        output_network = individual.get_output_network()
        unconnected = individual.get_num_unconnected_nodes()
        penalty = unconnected * args["penalty_scale"]
        penalty += 0 if individual.any_hidden_nodes() else args["penalty_scale"]
        start = time.time()
        scores = amp.ActualPool(25, "fitness").run(env.run_network, [output_network] * args["fitness_iterations"])
        score = np.average(scores)
        end = time.time()
        hedwig.info(f"individual {id:4} got fitness {score:8.2f} on network with {len(output_network.network.node[0]):4} nodes with time {end - start:8.3f}")
        individual.set_score(score, penalty)
        return score

    @staticmethod
    def fitness_nn(individual: Individual, env, **args):
        output_network = individual.get_output_network()
        scores = amp.ActualPool(25, "fitness").run(env.run_nn_network, [output_network] * args["fitness_iterations"])
        score = np.average(scores)
        individual.set_score(score)
        return score