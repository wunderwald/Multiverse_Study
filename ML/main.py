'''
Open for experimentation / TODO:

- expand weight ranges?
- randomize crossover method for each crossover (add this as CROSSOVER_METHOD='shuffle', give tendency towards 'abs')

Create (parallel) batch optimization (run a lot of optimizations simultaneously)
- Stop on convergence (i.e. 30 generations with same best fitness)
- Store individuals that are close to max fitness in database
'''

import numpy as np
import genetic as gen

# set hyper-parameters
POPULATION_SIZE = 100
MAX_NUM_GENERATIONS = 200
DISTANCE_METRIC = 'abs'
CROSSOVER_METHOD = 'shuffle'
MUTATION_RATE = .1
MUTATION_SCALE = .8
SELECT_PARENTS_METHOD = 'sus'
PARENT_RATIO = 0.1
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
    select_parents_method=SELECT_PARENTS_METHOD,
    parent_ratio=PARENT_RATIO,
    log=LOG,
    plot=PLOT
)
best_fitness = np.max(fitness)

# export data
# TODO write best individuals to database