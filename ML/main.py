'''
TODO
Create (parallel) batch optimization (run a lot of optimizations simultaneously)
- Stop on convergence (i.e. 30 generations with same best fitness)
- add a bit of hyperparameter randomness for each optimization (also population size)
- Store individuals that are close to max fitness in database
'''

from pymongo import MongoClient
from datetime import datetime
import numpy as np
import genetic as gen

# set hyper-parameters
POPULATION_SIZE = 200
MAX_NUM_GENERATIONS = 200
DISTANCE_METRIC = 'abs'
CROSSOVER_METHOD = 'shuffle'
MUTATION_RATE = .1
MUTATION_SCALE = .9
SELECT_PARENTS_METHOD = 'sus'
PARENT_RATIO = 0.1
TARGET_ZLC = 300.0
STOP_ON_CONVERGENCE = True

# Output parameters
WRITE_TO_DATABASE = True
LOG = True
PLOT = True

# initialize database
if WRITE_TO_DATABASE:
    # connect to mongo db client
    mongodb_client = MongoClient('mongodb://localhost:27017/')
    # open or create database
    db = mongodb_client['genetic_rsa']
    # open or create collection for current optimization batch
    db_collection = db[f"fittest_individuals_{int(round(datetime.now().timestamp() * 1000))}"]

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
    STOP_ON_CONVERGENCE=STOP_ON_CONVERGENCE,
    log=LOG,
    plot=PLOT
)
best_fitness = np.max(fitness)

# export data
if WRITE_TO_DATABASE:
    # collect hyperparameters
    hyperparameters = {
        'POPULATION_SIZE': POPULATION_SIZE,
        'MAX_NUM_GENERATIONS': MAX_NUM_GENERATIONS,
        'DISTANCE_METRIC': DISTANCE_METRIC,
        'CROSSOVER_METHOD': CROSSOVER_METHOD,
        'MUTATION_RATE': MUTATION_RATE,
        'MUTATION_SCALE': MUTATION_SCALE,
        'SELECT_PARENTS_METHOD': SELECT_PARENTS_METHOD,
        'PARENT_RATIO': PARENT_RATIO,
        'TARGET_ZLC': TARGET_ZLC,
        'STOP_ON_CONVERGENCE': STOP_ON_CONVERGENCE
    }
    # select fittest individuals (indivisuals in the best 10% of the fitness range)
    fitness_range = abs(np.max(fitness) - np.min(fitness))
    fittest_individuals = [{'individual': i, 'fitness': f} for i, f in zip(final_population, fitness) if f >= best_fitness - .1 * fitness_range]
    # make database record
    record = {
        'hyperparameters': hyperparameters,
        'fittest_individuals': fittest_individuals
    }
    # write record to database
    db_collection.insert_one(record)

    