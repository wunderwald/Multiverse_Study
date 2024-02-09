import numpy as np
import ibi_generator as ibi
import rsa_drew as rsa

# toggle logging
LOG = True

# ------------------------------
# CONSTANTS AND PARAMETER RANGES
# ------------------------------

# constants for ibi generation
RECORDING_TIME_S = 300
NUM_VLF_FREQS = 4
NUM_LF_FREQS = 6
NUM_HF_FREQS = 6
NUM_TOTAL_FREQS = 16

# parameter ranges for IBI generation
rng_base_ibi_adult = [650, 750]
rng_base_ibi_infant = [450, 550]
rng_freq_vlf = [0.01, 0.04]
rng_freq_lf = [0.04, 0.15]
rng_freq_hf = [0.15, 0.4]
rng_weights_vlf = [0.015, 1.0]
rng_weights_lf = [0.004, 0.4]
rng_weights_hf = [0.002, 0.2]
rng_phase_shift = [0, 2 * np.pi]

# constants for parameter optimization
TARGET_ZLC = 300

# ---------------------------
# GENETIC ALGORITHM FUNCTIONS
# ---------------------------

def initialize_individual():
    '''
    TODO: documentation
    '''
    # Randomize base inter-beat interval
    base_ibi_adult = np.random.uniform(*rng_base_ibi_adult)
    base_ibi_infant = np.random.uniform(*rng_base_ibi_infant)

    # Randomize frequencies within each range
    vlf_freqs_adult = np.random.uniform(*rng_freq_vlf, NUM_VLF_FREQS)
    lf_freqs_adult = np.random.uniform(*rng_freq_lf, NUM_LF_FREQS)
    hf_freqs_adult = np.random.uniform(*rng_freq_hf, NUM_HF_FREQS)
    vlf_freqs_infant = np.random.uniform(*rng_freq_vlf, NUM_VLF_FREQS)
    lf_freqs_infant = np.random.uniform(*rng_freq_lf, NUM_LF_FREQS)
    hf_freqs_infant = np.random.uniform(*rng_freq_hf, NUM_HF_FREQS)

    # Combine all frequencies
    frequencies_adult = np.concatenate((vlf_freqs_adult, lf_freqs_adult, hf_freqs_adult))
    frequencies_infant = np.concatenate((vlf_freqs_infant, lf_freqs_infant, hf_freqs_infant))

    # Assign weights to each band
    vlf_weights_adult = np.random.uniform(*rng_weights_vlf, NUM_VLF_FREQS)
    lf_weights_adult = np.random.uniform(*rng_weights_lf, NUM_LF_FREQS)
    hf_weights_adult = np.random.uniform(*rng_weights_hf, NUM_HF_FREQS)
    vlf_weights_infant = np.random.uniform(*rng_weights_vlf, NUM_VLF_FREQS)
    lf_weights_infant = np.random.uniform(*rng_weights_lf, NUM_LF_FREQS)
    hf_weights_infant = np.random.uniform(*rng_weights_hf, NUM_HF_FREQS)

    # Combine all weights
    weights_adult = np.concatenate((vlf_weights_adult, lf_weights_adult, hf_weights_adult))
    weights_infant = np.concatenate((vlf_weights_infant, lf_weights_infant, hf_weights_infant))

    # Generate random phase shifts for each band
    phase_shifts_adult = np.random.uniform(0, 2 * np.pi, NUM_TOTAL_FREQS)
    phase_shifts_infant = np.random.uniform(0, 2 * np.pi, NUM_TOTAL_FREQS)

    # Collect parameters in dict
    individual = {}
    individual['base_ibi_adult'] = base_ibi_adult
    individual['base_ibi_infant'] = base_ibi_infant
    for i in range(NUM_TOTAL_FREQS):
        individual[f"freq_{i}_adult"] = frequencies_adult[i]
        individual[f"weight_{i}_adult"] = weights_adult[i]
        individual[f"phase_{i}_adult"] = phase_shifts_adult[i]
        individual[f"freq_{i}_infant"] = frequencies_infant[i]
        individual[f"weight_{i}_infant"] = weights_infant[i]
        individual[f"phase_{i}_infant"] = phase_shifts_infant[i]

    return individual

def initialize_population(population_size: int):
    '''
    TODO: documentation
    '''
    population = [initialize_individual() for _ in range(population_size)]
    return np.array(population)
    

def extract_ibi_params(individual: dict):
    '''
    TODO: documentation
    '''
    adult_params = {
        'base_ibi': individual['base_ibi_adult'],
        'frequencies': [value for key, value in individual.items() if 'freq' in key and 'adult' in key],
        'freq_weights': [value for key, value in individual.items() if 'weight' in key and 'adult' in key],
        'phase_shifts': [value for key, value in individual.items() if 'phase' in key and 'adult' in key]
    }

    infant_params = {
        'base_ibi': individual['base_ibi_infant'],
        'frequencies': [value for key, value in individual.items() if 'freq' in key and 'infant' in key],
        'freq_weights': [value for key, value in individual.items() if 'weight' in key and 'infant' in key],
        'phase_shifts': [value for key, value in individual.items() if 'phase' in key and 'infant' in key]
    }

    return adult_params, infant_params



def evaluate_fitness_individual(individual: dict, distance_metric: str='euclidian'):
    '''
    Calculating RSA synchrony measured as the zero-lag coefficient (zlc) of RSA cross-correlation.
    Fitness is the deviation of the measured zcl from the target zlc based on the selected distance metric.

    Parameters:
    - individual (dict): key-value pairs for the 98 parameters for dyad IBI generator (see README for details)
    - distance_metric (str): distance metric for calculating difference between measured and optimal zlc (currently supported: 'euclidian')

    Returns:
    - fitness (float): the absolute difference between calculated and target ZLC, float('inf') on exception
    '''

    # extract ibi parameters
    adult_params, infant_params = extract_ibi_params(individual=individual)
    
    try:
        # generate IBIs
        adult_ibi, infant_ibi = ibi.generate_dyad_ibi(
            recording_time_s=RECORDING_TIME_S, 
            adult_params=adult_params, 
            infant_params=infant_params
        )

        # calculate synchrony
        zlc, _ = rsa.rsa_synchrony(adult_ibi, infant_ibi)

        # calculate fitness
        match distance_metric:
            # euclidian distance
            case 'euclidian':
                return abs(zlc - TARGET_ZLC)
            # default case (euclidian)
            case _:
                return abs(zlc - TARGET_ZLC)
    
    # return infinity on exception
    except ValueError:
        return float('inf')
    
def evaluate_fitness(population: np.array, distance_metric: str='euclidian'):
    '''
    Evaluate the fitness of the whole population (using deviation from ideal zero-lag coefficient of RSA cross-correlation as metric).
    
    Parameters:
    - population (np.array): the current population represented as an array of parameter dicts
    - distance_metric (str): distance metric for calculating difference between measured and target zlc (currently supported: 'euclidian')

    Returns:
    - fitness (np.array): fitness value for each individual
    '''
    return [evaluate_fitness_individual(individual, distance_metric) for individual in population]
    
def select_parents(population: np.array, fitness: np.array, num_parents: int):
    '''
    TODO: documentation
    '''
    return

def crossover(parents: np.array, offspring_size: np.array):
    '''
    TODO: documentation
    '''
    # create random key order for each generation before crossover
    return

def mutation(offspring_crossover: np.array):
    '''
    TODO: documentation
    '''
    return

def succession(population: np.array, distance_metric: str, crossover_method: str):
    '''
    TODO: documentation
    '''

    new_population = None
    
    return new_population


# main function: run genetic evolution
def evolution(population_size: int, max_num_generations: int, fitness_thresh: float, distance_metric: str, crossover_method: str):
    '''
    TODO: documentation
    '''
    
    # initialize population and fitness
    population = initialize_population(population_size)
    fitness = evaluate_fitness(population, distance_metric)

    # iterate over generations
    for generation_index in range(max_num_generations):

        # create new generation of population
        population = succession(population, distance_metric, crossover_method)

        # calculate fitness
        fitness = evaluate_fitness(population)
        best_fitness = np.min(fitness)        

        # inform about fitness state
        if LOG:
            print(f"# Generation {generation_index} - best fitness: {best_fitness}")

        # terminate evolution if desired fitness is reached
        if best_fitness < fitness_thresh:
            break

    return zip(population, fitness)
