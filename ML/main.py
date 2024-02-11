'''
Open for experimentation / TODO:
    
- expand weight ranges
- randomize crossover method for each crossover
- add parrallelization: evaluate individuals
- apply threshold not to min (bc thats probably an outlier) but to quantile or smth
- specialise mutation to parameter characteristics
- work on exploration / exploitation balance
'''

import numpy as np
import genetic as gen

# set hyper-parameters
POPULATION_SIZE = 200
MAX_NUM_GENERATIONS = 50
DISTANCE_METRIC = 'abs'
CROSSOVER_METHOD = 'arithmetic'
MUTATION_RATE = .1
MUTATION_SCALE = .05
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
    log=LOG,
    plot=PLOT
)
best_fitness = np.max(fitness)

# export data
# TODO write best individuals to database