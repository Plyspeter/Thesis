from inspect import getmembers, isfunction
from evolution.individual import Individual

class Crossover:

    @staticmethod
    def run(method, parent1, parent2, args):
        func = Crossover.get_function(method)
        return func(parent1, parent2, **args)

    @staticmethod
    def crossover(parent1 : Individual, parent2 : Individual):
        raise NotImplementedError
    
    @staticmethod
    def get_function(name):
        for func_name, func in getmembers(Crossover, isfunction):
            if func_name == name:
                return func