'''
### NEXT STEPS ###

- [initialize_individual: replace dics by np.arrays (facilitates actually everything a LOT)]
- therefore: upate extract_ibi_params / create param_dict_to_array and param_array_to_dict
- implement crossover functions (using array params)
- implement mutation function (using array params)
- implement log difference (and further methods)
- implement apply_limits (using array params, also create ranges to array_of_ranges function)
- add documentation
'''

import numpy as np
import genetic as gen

# set hyper-parameters
POPULATION_SIZE = 1000
MAX_NUM_GENERATIONS = 100
FITNESS_THRESH = .01
DISTANCE_METRIC = 'abs'
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
- expand weight ranges
'''