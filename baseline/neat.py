import numpy as np
import prettyNEAT
import domain
import actual_multiprocessing.actual_pool as pool

from baseline.neat_games.neat_game_cart_pole import NeatGameCartPole
from baseline.neat_games.neat_game_pendulum import NeatGamePendulum
from baseline.neat_games.neat_game_acrobot import NeatGameAcroBot
from baseline.neat_games.neat_game_mountaincar import NeatGameMountainCar
from baseline.neat_games.neat_game_bipedalwalker import NeatGameBipedalWalker
from baseline.neat_games.neat_game_lunarlander import NeatGameLunarLander

# This class is a baseline test on how neat solves different problems
# It has been setup using the prettyNEAT library: https://github.com/d9w/prettyNEAT
# It works directly on gym envoronments, but does not have all of them implemented by default
# So remember the configurations to the game list as shown in the neat_games folder

def func(ind):
    task = domain.GymTask(domain.games['BipedalWalker'], nReps=100)
    wVec = ind.wMat.flatten()
    aVec = ind.aVec.flatten()
    return task.getFitness(wVec, aVec)

class Neat:

    def run(self, fitness_iterations, gym_env_name, render=False):
        # Gym_env_name needs to be set in the default_neat.json file!
        NeatGameCartPole().add()
        NeatGamePendulum().add()
        NeatGameAcroBot().add()
        NeatGameMountainCar().add()
        NeatGameLunarLander().add()
        NeatGameBipedalWalker().add()

        hyp = domain.loadHyp(pFileName='baseline/neat_games/baseline_neat.json')
        network = prettyNEAT.Neat(hyp)

        best_reward = -2000
        itter_without_improvement = 0

        p = pool.ActualPool(20, 'NEAT_BASELINE')

        while itter_without_improvement < 40:
            pop = network.ask()
            reward = p.run(func, pop)

            if abs(best_reward - np.max(reward)) < 1.0:
                best_reward = np.max(reward)
                itter_without_improvement = -1
            
            network.tell(reward)
            itter_without_improvement += 1
            p.reset()

        return (np.average(reward), np.max(reward))