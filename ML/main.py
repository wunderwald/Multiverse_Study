from multiprocessing import Pool
from pymongo import MongoClient
from datetime import datetime
import numpy as np
import genetic as gen
import hyperparameters as hyper

# genetic parameter optimization
def genetic_optimization(i):

    # set hyper-constants
    TARGET_ZLC=300
    MAX_NUM_GENERATIONS = 200
    DISTANCE_METRIC = 'euclidean'
    STOP_ON_CONVERGENCE = True
    CONVERGENCE_N = 30
    RANDOM_HYPERPARAMETERS = True

    # get hyper-parameters
    hyperparams = hyper.random_hyperparams() if RANDOM_HYPERPARAMETERS else hyper.default_hyperparams()

    # Output parameters
    WRITE_TO_DATABASE = True
    LOG = True
    LOG_MINIMAL = True
    PLOT = False

    # run genetic evolution algorithm
    final_population, fitness, last_generation_index = gen.evolution(
        max_num_generations=MAX_NUM_GENERATIONS,
        distance_metric=DISTANCE_METRIC,
        target_zlc=TARGET_ZLC,
        stop_on_convergence=STOP_ON_CONVERGENCE,
        convergence_N=CONVERGENCE_N,
        population_size=hyperparams['POPULATION_SIZE'],        
        crossover_method=hyperparams['CROSSOVER_METHOD'],
        mutation_rate=hyperparams['MUTATION_RATE'],
        mutation_scale=hyperparams['MUTATION_SCALE'],
        select_parents_method=hyperparams['SELECT_PARENTS_METHOD'],
        parent_ratio=hyperparams['PARENT_RATIO'],
        
        log=LOG,
        plot=PLOT
    )
    best_fitness = np.max(fitness)

    # log
    if LOG_MINIMAL:
        print(f'# optimization {i} done, best fitness: {best_fitness}, num generations: {last_generation_index}')

    # export data
    if WRITE_TO_DATABASE:
        
        # connect to mongo db client
        mongodb_client = MongoClient('mongodb://localhost:27017/')

        # open or create database
        db = mongodb_client['genetic_rsa']

        # open or create collection for current optimization batch
        db_collection = db[f"fittest_individuals_{int(round(datetime.now().timestamp() * 1000))}"]

        # collect hyperparameters and constants
        hyperparams_and_constants = {
            'MAX_NUM_GENERATIONS': MAX_NUM_GENERATIONS,
            'DISTANCE_METRIC': DISTANCE_METRIC,
            'TARGET_ZLC': TARGET_ZLC,
            'STOP_ON_CONVERGENCE': STOP_ON_CONVERGENCE,
            'CONVERGENCE_N': CONVERGENCE_N,
            'POPULATION_SIZE': hyperparams['POPULATION_SIZE'],
            'CROSSOVER_METHOD': hyperparams['CROSSOVER_METHOD'],
            'MUTATION_RATE': hyperparams['MUTATION_RATE'],
            'MUTATION_SCALE': hyperparams['MUTATION_SCALE'],
            'SELECT_PARENTS_METHOD': hyperparams['SELECT_PARENTS_METHOD'],
            'PARENT_RATIO': hyperparams['PARENT_RATIO'],     
        }

        # select fittest individuals (indivisuals in the best 20% of the fitness range)
        fitness_range = abs(np.max(fitness) - np.min(fitness))
        fittest_individuals = [{'individual': i, 'fitness': f} for i, f in zip(final_population, fitness) if f >= best_fitness - .2 * fitness_range or f > 50]
        
        # make database record
        record = {
            'hyperparameters': hyperparams_and_constants,
            'fittest_individuals': fittest_individuals
        }

        # write record to database
        db_collection.insert_one(record)

# execute batch of optimizations in parallel
if __name__ == '__main__':
    NUM_PARALLEL_OPTIMIZATIONS = 500
    with Pool() as pool:
        pool.map(genetic_optimization, range(NUM_PARALLEL_OPTIMIZATIONS))