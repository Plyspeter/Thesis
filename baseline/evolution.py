from evolution.individual_generator import IndividualGenerator
from gym_env.gym_env import GymEnv
from evolution.evolution import Evolution as Evo
import hedwig

class Evolution:

    def run(self, fitness_iterations, gym_env_name):
        env = GymEnv(gym_env_name)

        hidden_layer_size = round(env.get_input_size() * 0.66)
        
        ind_gen = IndividualGenerator(None, None)
        ind_config = {
            "kind": "nn",
            "nca_topology": {
                "input_size": env.get_input_size(), 
                "output_size": env.get_output_size(), 
                "hidden_layer_sizes": [hidden_layer_size, hidden_layer_size, hidden_layer_size],
                "acts": ["tanh", "tanh", "tanh", "tanh"]
            },
            "growth_iterations": 4
        }
        ind_gen.set_parameters(**ind_config)

        evo_config = {
            "pop_size": 300,
            "num_of_parents": 10,
            "random_parent_prob": 0.20,

            "fitness": {
                "func": "fitness_nn",
                "vars": {
                    "fitness_iterations": fitness_iterations,
                    "penalty_scale": 1000,
                    "env_kind": "gym",
                    "env_name": gym_env_name
                }
            },

            "mutation": {
                "func": "gauss_mutate",
                "vars": {
                    "num_of_mutations": 10,
                    "num_of_changes": 1,
                    "gauss_scale": 1.0,
                    "weight_mutation_range": 2.5,
                    "bias_mutation_range": 2.5,
                    "bias_proc_chance": 0.25
                }
            }
        }

        evolution_sim = Evo(ind_gen)
        evolution_sim.set_parameters(**evo_config)
        evolution_sim.run_evolution(callback)

        score = 0
        for ind in evolution_sim.get_population():
            score += ind.get_score()

        score /= evo_config['pop_size']
        best_score = evolution_sim.get_best_pop().get_score()

        return (score, best_score)

def callback(iteration, iter_no_improve, population):
    if iteration % 20 == 0:
        hedwig.info(f'Iteration reached: {iteration}')
    if iter_no_improve % 10 == 0:
        hedwig.info(f'No improvement in {iter_no_improve} generations')

    return iter_no_improve >= 40
