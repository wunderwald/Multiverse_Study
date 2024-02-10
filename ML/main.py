import numpy as np
import genetic as gen

# set hyper-parameters
POPULATION_SIZE = 80
MAX_NUM_GENERATIONS = 100
FITNESS_THRESH = 0.5
DISTANCE_METRIC = 'log'
CROSSOVER_METHOD = 'blend'
MUTATION_RATE = .2
MUTATION_SCALE = .15
TARGET_ZLC = 300.0

# Output parameters
LOG = True
PLOT = True

# run genetic evolution algorithm
final_population, fitness = gen.evolution(
    population_size=POPULATION_SIZE,
    max_num_generations=MAX_NUM_GENERATIONS,
    fitness_thresh=FITNESS_THRESH,
    target_zlc=TARGET_ZLC,
    distance_metric=DISTANCE_METRIC,
    crossover_method=CROSSOVER_METHOD,
    mutation_rate=MUTATION_RATE,
    mutation_scale=MUTATION_SCALE,
    log=LOG,
    plot=PLOT
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
- apply threshold not to min (bc thats probably an outlier) but to quantile or smth
- specialise mutation to parameter characteristics
'''