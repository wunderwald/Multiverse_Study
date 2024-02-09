import numpy as np
import genetic as gen

# set hyper-parameters
POPULATION_SIZE = 1000
MAX_NUM_GENERATIONS = 100
FITNESS_THRESH = .01
DISTANCE_METRIC = 'euclidian'
CROSSOVER_METHOD = None
TARGET_ZLC = 300.0

# run genetic evolution algorithm
final_population, fitness = gen.evolution(
    population_size=POPULATION_SIZE,
    max_num_generations=MAX_NUM_GENERATIONS,
    fitness_thresh=FITNESS_THRESH,
    target_zlc=TARGET_ZLC,
    distance_metric=DISTANCE_METRIC,
    crossover_method=CROSSOVER_METHOD
)
best_fitness = np.min(fitness)

# export data
# TODO write best individuals to database


'''
Open for experimentation:
- num_parents / offspring_size (for now, both equal to population_size//2)
    -> introduce more variation by increasing offspring to parents ratio
'''