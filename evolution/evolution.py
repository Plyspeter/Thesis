from copy import copy, deepcopy
from random import Random
import time

import numpy
from evolution.crossover import Crossover
from evolution.evolution_interface import EvolutionInterface
from evolution.fitness import Fitness
from evolution.mutation import Mutation
import hedwig
import actual_multiprocessing as amp
        
class Evolution(EvolutionInterface):

    rand = Random()

    def __init__(self, individual_generator) -> None:
        self.__cross = None
        self.__mutation = None
        self.__fitness = None
        self.__individual_generator = individual_generator
    
    def set_parameters(self, **config):
        hedwig.debug("Setting parameters for Evolution")
        for key, val in config.items():
            setattr(self, "_Evolution__" + key, val)
        hedwig.debug("Parameters set")

    def run_evolution(self, callback):
        hedwig.debug("Evolution begun")
        if self.__fitness is None:
            raise Exception("Fitness vars must be given to SimpleEvolution")
        
        pop = [self.__individual_generator.get_individual() for _ in range(self.__pop_size)]
        
        start = time.time()
        self.__pop = self.__get_fitness(pop)
        end = time.time()
        hedwig.debug(f"Initial population fitness took: {end - start}")
        self.__pop = sorted(self.__pop, key=lambda ind: ind.get_score(), reverse=True)
        
        current_best = self.__pop[0]

        iterations = 0
        iterations_wo_improvement = 0
        while not callback(iterations, iterations_wo_improvement, self.__pop):
            parents = self.__parent_selection(self.__pop)
            offspring = self.__generate_offspring(parents)
            
            start = time.time()
            newIndividuals_fits = self.__get_fitness(offspring)
            end = time.time()
            hedwig.debug(f"Children fitness took: {end - start}")
            
            self.__pop = sorted(newIndividuals_fits + self.__pop, key=lambda ind: ind.get_score(), reverse=True)
            self.__pop = self.__survivor_selection(self.__pop)
            
            if abs(current_best.get_score() - self.__pop[0].get_score()) < 2.0:
                iterations_wo_improvement += 1
            else:
                current_best = self.__pop[0]
                iterations_wo_improvement = 0
            iterations += 1
        hedwig.debug("Evolution ended")
        
        scores = [ind.get_score() for ind in self.__pop]
        return self.__pop

    def get_population(self):
        return self.__pop
    
    def get_best_pop(self):
        return self.__pop[0]
    
    def __parent_selection(self, fits_pop):
        hedwig.debug("Selecting parents")
        parents = []
        i = 0
        for _ in range(self.__num_of_parents):
            if Evolution.rand.random() < self.__random_parent_prob:
                parents.append(fits_pop[Evolution.rand.randint(0, len(fits_pop) - 1)])
            else:
                parents.append(fits_pop[i])
                i += 1
        hedwig.debug("Parents selected")
        return parents

    def __get_fitness(self, pop):
        hedwig.debug(f"Fitness calculation begun for batch with size: {len(pop)}")
        func = self.__fitness["func"]
        vars = self.__fitness["vars"]

        def fit_func(individual) -> None:
            return Fitness.run(func, individual, vars)
        
        parallel = amp.ActualMultiprocessing(10, "evo", fit_func, pop)
        results = parallel.run()
        
        return [ind for _, ind in results]
        
    def __generate_offspring(self, parents):
        hedwig.debug("Generating offspring")
        mutations = []
        children = []
        if(self.__mutation is not None):
            mutation_list = list(map(lambda parent: Mutation.run(self.__mutation["func"], parent, self.__mutation["vars"]), parents))
            mutations = [mutation for mutations in mutation_list for mutation in mutations]
        if(self.__cross is not None):
            for i in range(len(parents)-1):
                children.append(Crossover.run(self.__cross["func"], parents[i], parents[i+1], self.__cross["vars"]))
                
        offspring = mutations + children
        if len(offspring) == 0:
            raise Exception("No offspring generated in Evolution")
        hedwig.debug("Offspring generated")
        return offspring
    
    def __survivor_selection(self, pop):
        hedwig.debug("Selecting survivors")
        survivors = pop[:self.__pop_size]
        hedwig.debug("Survivors selected")
        return survivors
    
