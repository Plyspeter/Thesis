import sys
import os

if len(sys.argv) > 1 and sys.argv[1] == "HPC":
    os.chdir(os.getcwd() + "/thesis")

import baseline as bl
import hedwig
import numpy as np

hedwig.init("logging_config.json")
# This class can run different baseline algorithms
# That can be used for comparisons

# List of parameters used for baseline testing:
# How many times the baseline algorithm is run to compare averages
run_iterations = 14

# How many times the gym environment is run to 
# reduce the impact of random seeds on the result
fitness_iterations = 100

# The name of the gym environment, that the baseline
# is tested on
#gym_env_name = "CartPole-v1"
#gym_env_name = "Acrobot-v1"
#gym_env_name = "Pendulum-v1"
#gym_env_name = "MountainCar-v0"
gym_env_name = "BipedalWalker-v3"
#gym_env_name = "LunarLander-v2"

# List of different available baseline algorithms
#baseline = bl.Random()

#baseline = bl.RandomSearch()

baseline = bl.Evolution()

#baseline = bl.Neat()

# Running the choosen baseline algorithm
results_avg = []
results_best = []
with open(f'baseline_evo_{gym_env_name[:-3]}.txt', 'w') as f:
    for i in range(run_iterations):
        hedwig.info(f'Baseline is {i} of {run_iterations} complete')
        hedwig.info(f'Baseline type: {type(baseline)} --- Baseline env: {gym_env_name}')
        avg, best = baseline.run(fitness_iterations, gym_env_name)  
        f.write(str(avg) + ',' + str(best) + '\n')
        results_avg.append(avg)
        results_best.append(best)
        hedwig.info(f'Run completed, avg: {avg} --- best: {best}')

# Outputing the results of the baseline algorithm
hedwig.info(f'Baseline has finished running!')
hedwig.info(f'Baseline average average: {np.average(results_avg)}')
hedwig.info(f'Baseline average best: {np.max(results_avg)}')
hedwig.info(f'Baseline average standard deviation: {np.std(results_avg)}')
hedwig.info(f'------------------------------------------------------------')
hedwig.info(f'Baseline best average: {np.average(results_best)}')
hedwig.info(f'Baseline best best: {np.max(results_best)}')
hedwig.info(f'Baseline best standard deviation: {np.std(results_best)}')

hedwig.critical_info(f'Baseline completed: {gym_env_name}')