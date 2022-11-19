import torch
from evolution.individual import Individual
from numpy.random import normal
import random
from inspect import getmembers, isfunction

class Mutation:

    @staticmethod
    def run(method, individual, args):
        func = Mutation.get_function(method)
        return func(individual, **args)

    @staticmethod
    def mutate(individual : Individual, **mutation):
        new_individuals = []
        for _ in range(mutation["num_of_mutations"]):
            new_individual = individual.copy_for_mutation()
            layer_id = random.randint(0, new_individual.get_num_of_layers() - 1)
            neuron_id = random.randint(0, new_individual.get_num_of_neurons(layer_id) - 1)
            
            if random.random() > mutation["bias_proc_chance"]:
                weight_id = random.randint(0, new_individual.get_num_of_weights(layer_id, neuron_id) - 1)
                new_individual.add_to_weight(layer_id, neuron_id, weight_id, random.random() * mutation["weight_mutation_range"] - mutation["weight_mutation_range"]/2)
            else:
                new_individual.add_to_bias(layer_id, neuron_id, random.random() * mutation["bias_mutation_range"] - mutation["bias_mutation_range"]/2)
            new_individuals.append(new_individual)            
        return new_individuals


    @staticmethod
    def gauss_mutate(individual : Individual, **mutation):
        new_individuals = []
        
        for _ in range(mutation["num_of_mutations"]):
            new_individual = individual.copy_for_mutation()
            for _ in range(mutation["num_of_changes"]):
                
                #Randomly pick a layer proportional to the size of each layer
                layer_id = new_individual.get_random_layer()

                #Randomly pick a neuron in the chosen layer
                neuron_id = random.randint(0, new_individual.get_num_of_neurons(layer_id) - 1)

                #Sample for Gaussian distribution
                noise = torch.tensor(normal(loc=0.0, scale=mutation["gauss_scale"]))

                #Randomly choose to mutate bias or weight proportional to the amount of weight
                if random.random() > (1 / new_individual.get_num_of_weights(layer_id, neuron_id)):
                    weight_id = random.randint(0, new_individual.get_num_of_weights(layer_id, neuron_id) - 1)
                    new_individual.add_to_weight(layer_id, neuron_id, weight_id, noise)
                else:
                    new_individual.add_to_bias(layer_id, neuron_id, noise)
                    
                new_individuals.append(new_individual)    
        return new_individuals
    
    @staticmethod
    def get_function(name):
        for func_name, func in getmembers(Mutation, isfunction):
            if func_name == name:
                return func
    
