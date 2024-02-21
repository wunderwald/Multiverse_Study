'''
Open for experimentation / TODO:
    
- algorithm converges quickly now and results in a 50/50 split of good and bad results. 
    -> work on exploraion/exploitation balance
    -> experiment with increasing offspring/old_population ratio
    -> add more mutation (mb not only to offspring)
    -> increase weight ranges

- expand weight ranges?
- randomize crossover method for each crossover
- add parrallelization: evaluate individuals
- specialise mutation to parameter characteristics
'''

import numpy as np
import genetic as gen

# set hyper-parameters
POPULATION_SIZE = 200
MAX_NUM_GENERATIONS = 50
DISTANCE_METRIC = 'abs'
CROSSOVER_METHOD = 'arithmetic'
MUTATION_RATE = .3
MUTATION_SCALE = .7
PARENT_RATIO = 0.7
TARGET_ZLC = 300.0

# Output parameters
LOG = True
PLOT = True

# run genetic evolution algorithm
final_population, fitness = gen.evolution(
    population_size=POPULATION_SIZE,
    max_num_generations=MAX_NUM_GENERATIONS,
    target_zlc=TARGET_ZLC,
    distance_metric=DISTANCE_METRIC,
    crossover_method=CROSSOVER_METHOD,
    mutation_rate=MUTATION_RATE,
    mutation_scale=MUTATION_SCALE,
    parent_ratio=PARENT_RATIO,
    log=LOG,
    plot=PLOT
)
best_fitness = np.max(fitness)

# export data
# TODO write best individuals to database