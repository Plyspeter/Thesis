import hedwig
import numpy as np

hedwig.init("logging_config.json")

results_avg = []
results_best = []

with open(f'baseline_evo_LunarLander.txt') as f:
    for line in f.readlines():
        x = line[:-2].split(',')
        results_avg.append(float(x[0]))
        results_best.append(float(x[1]))

hedwig.info(f'Reading baseline results!')
hedwig.info(f'Baseline average average: {np.average(results_avg)}')
hedwig.info(f'Baseline average best: {np.max(results_avg)}')
hedwig.info(f'Baseline average standard deviation: {np.std(results_avg)}')
hedwig.info(f'------------------------------------------------------------')
hedwig.info(f'Baseline best average: {np.average(results_best)}')
hedwig.info(f'Baseline best best: {np.max(results_best)}')
hedwig.info(f'Baseline best standard deviation: {np.std(results_best)}')
