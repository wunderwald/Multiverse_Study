import numpy as np
import genetic as gen

# set hyper-parameters
POPULATION_SIZE = 160
MAX_NUM_GENERATIONS = 100
FITNESS_THRESH = 0.3
DISTANCE_METRIC = 'log'
CROSSOVER_METHOD = None
MUTATION_RATE = .1
MUTATION_SCALE = .1
TARGET_ZLC = 300.0

# run genetic evolution algorithm
final_population, fitness = gen.evolution(
    population_size=POPULATION_SIZE,
    max_num_generations=MAX_NUM_GENERATIONS,
    fitness_thresh=FITNESS_THRESH,
    target_zlc=TARGET_ZLC,
    distance_metric=DISTANCE_METRIC,
    crossover_method=CROSSOVER_METHOD,
    mutation_rate=MUTATION_RATE,
    mutation_scale=MUTATION_SCALE
)
best_fitness = np.min(fitness)

# export data
# TODO write best individuals to database

'''
Open for experimentation:
- num_parents / offspring_size (for now, both equal to population_size//2)
    -> introduce more variation by increasing offspring to parents ratio
- expand weight ranges
- randomize crossover method for each crossover
- add parrallelization: evaluate individuals
'''