from multiprocessing import Pool
from pymongo import MongoClient
import numpy as np
import genetic as gen
import hyperparameters as hyper

# genetic parameter optimization
def genetic_optimization(args):

    # parse args
    optimization_index = args['optimization_index']
    ibi_range_type = args['ibi_range_type']
    db_name = args['db_name']
    db_collection = args['db_collection']
    use_noise = args['use_noise']

    # set hyper-constants
    TARGET_ZLC=300
    MAX_NUM_GENERATIONS = 200
    DISTANCE_METRIC = 'euclidean'
    STOP_ON_CONVERGENCE = True
    CONVERGENCE_N = 30
    USE_NOISE = use_noise
    RANDOM_HYPERPARAMETERS = True

    # get hyper-parameters
    hyperparams = hyper.random_hyperparams() if RANDOM_HYPERPARAMETERS else hyper.default_hyperparams()

    # Output parameters
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
        ibi_range_type=ibi_range_type,
        use_noise = USE_NOISE,
        log=LOG,
        plot=PLOT
    )
    best_fitness = np.max(fitness)

    # log
    if LOG_MINIMAL:
        print(f'# optimization {optimization_index} done, best fitness: {best_fitness}, num generations: {last_generation_index}')
  
    # connect to mongo db client
    mongodb_client = MongoClient('mongodb://localhost:27017/')

    # open or create database
    db = mongodb_client[db_name]

    # open or create collection for current optimization batch
    db_collection = db[db_collection]

    # collect hyperparameters and constants
    hyperparams_and_constants = {
        'MAX_NUM_GENERATIONS': MAX_NUM_GENERATIONS,
        'DISTANCE_METRIC': DISTANCE_METRIC,
        'TARGET_ZLC': TARGET_ZLC,
        'USE_NOISE': USE_NOISE,
        'STOP_ON_CONVERGENCE': STOP_ON_CONVERGENCE,
        'CONVERGENCE_N': CONVERGENCE_N,
        'POPULATION_SIZE': hyperparams['POPULATION_SIZE'],
        'CROSSOVER_METHOD': hyperparams['CROSSOVER_METHOD'],
        'MUTATION_RATE': hyperparams['MUTATION_RATE'],
        'MUTATION_SCALE': hyperparams['MUTATION_SCALE'],
        'SELECT_PARENTS_METHOD': hyperparams['SELECT_PARENTS_METHOD'],
        'PARENT_RATIO': hyperparams['PARENT_RATIO'],  
        'IBI_BASE_RANGE_TYPE': ibi_range_type   
    }

    # select fittest individuals (indivisuals in the best 20% of the fitness range or with fitness larger than 50)
    fitness_range = abs(np.max(fitness) - np.min(fitness))
    fittest_individuals = [{'individual': i, 'fitness': f} for i, f in zip(final_population, fitness) if f >= best_fitness - .2 * fitness_range or f > 50]
    
    # make database record
    record = {
        'hyperparameters': hyperparams_and_constants,
        'fittest_individuals': fittest_individuals
    }

    # write record to database
    db_collection.insert_one(record)

# batch genetic optimization
def genetic_optimization_batch(db_name, db_collection, use_noise=False, ibi_range_type='physiological', num_optimizations=200):
    static_params = {'db_name': db_name, 'db_collection': db_collection, 'use_noise': use_noise, 'ibi_range_type': ibi_range_type}
    params = [{**static_params, 'optimization_index': i} for i in range(num_optimizations)]
    with Pool() as pool:
        pool.map(genetic_optimization, params)

def run_gen_batches_mixed_params(db_name, num_optimizations):
    for use_noise in [False, True]:
        for ibi_range_type in ['physiological', 'extended_separated', 'extended_overlapping', 'extended_equal']:
            db_collection = f"genetic__{ibi_range_type}_ibi_{'w_noise' if use_noise else ''}"
            print(f"##### Running genetic optimization batch {'with' if use_noise else 'without'} noise with ibi_range_type={ibi_range_type} #####")
            genetic_optimization_batch(
                db_name=db_name, 
                db_collection=db_collection,
                use_noise=use_noise,
                ibi_range_type=ibi_range_type,
                num_optimizations=num_optimizations
            )

def run_brute_force_mixed_params(db_name, num_results):
    for use_noise in [False, True]:
        for ibi_range_type in ['physiological', 'extended_separated', 'extended_overlapping', 'extended_equal']:
            print(f"##### Running brute force {'with' if use_noise else 'without'} noise with ibi_range_type={ibi_range_type} #####")
            db_collection = f"brute_force__{ibi_range_type}_ibi_{'w_noise' if use_noise else ''}"

            gen.brute_force(
                target_zlc=200, 
                max_deviation=100, 
                num_results=1000, 
                use_noise=use_noise, 
                ibi_range_type=ibi_range_type, 
                log=True
            )

if __name__ == '__main__':
    DB_NAME = 'genetic_x_brute_force'
    # run brute force batches
    #TODO Brute
    # run genetic batches
    run_gen_batches_mixed_params(db_name=DB_NAME, num_optimizations=500)