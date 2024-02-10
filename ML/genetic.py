import numpy as np
import ibi_generator as ibi
import rsa_drew as rsa

# -------
# GLOBALS
# -------

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

# -------
# HELPERS
# -------

def clamp(val: float, min: float, max: float):
    return min if val < min else (max if max < val else val)

def get_limits(key: str):
    if 'phase' in key:
        return [float('-inf'), float('inf')]
    if 'ibi' in key and 'adult' in key:
        return rng_base_ibi_adult
    if 'ibi' in key and 'infant' in key:
        return rng_base_ibi_infant
    if 'weight' in key and 'vlf' in key:
        return rng_weights_vlf
    if 'weight' in key and 'lf' in key:
        return rng_weights_lf
    if 'weight' in key and 'hf' in key:
        return rng_weights_hf
    if 'freq' in key and 'vlf' in key:
        return rng_freq_vlf
    if 'freq' in key and 'lf' in key:
        return rng_freq_lf
    if 'freq' in key and 'hf' in key:
        return rng_freq_hf
    return [float('-inf'), float('inf')]

def apply_limits(individual: dict):
    '''
    Makes sure that all parameters / genes are in their corresponding limits

    Parameters:
    - individual (dict): an individual

    Returns:
    - individual_clamped (dict): an individual with all parameter values safely inside the corresponding limits
    '''
    individual_clamped = {}
    for key, value in individual.items():
        limits = get_limits(key)
        individual_clamped[key] = clamp(value, *limits)
    return individual_clamped

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

# ---------------------------
# GENETIC ALGORITHM FUNCTIONS
# ---------------------------

def initialize_individual():
    '''
    TODO: documentation
    '''
    # Randomize base inter-beat interval
    base_ibi_adult = int(np.random.uniform(*rng_base_ibi_adult))
    base_ibi_infant = int(np.random.uniform(*rng_base_ibi_infant))

    # Randomize frequencies within each range
    vlf_freqs_adult = np.random.uniform(*rng_freq_vlf, NUM_VLF_FREQS)
    lf_freqs_adult = np.random.uniform(*rng_freq_lf, NUM_LF_FREQS)
    hf_freqs_adult = np.random.uniform(*rng_freq_hf, NUM_HF_FREQS)
    vlf_freqs_infant = np.random.uniform(*rng_freq_vlf, NUM_VLF_FREQS)
    lf_freqs_infant = np.random.uniform(*rng_freq_lf, NUM_LF_FREQS)
    hf_freqs_infant = np.random.uniform(*rng_freq_hf, NUM_HF_FREQS)

    # Assign weights to each band
    vlf_weights_adult = np.random.uniform(*rng_weights_vlf, NUM_VLF_FREQS)
    lf_weights_adult = np.random.uniform(*rng_weights_lf, NUM_LF_FREQS)
    hf_weights_adult = np.random.uniform(*rng_weights_hf, NUM_HF_FREQS)
    vlf_weights_infant = np.random.uniform(*rng_weights_vlf, NUM_VLF_FREQS)
    lf_weights_infant = np.random.uniform(*rng_weights_lf, NUM_LF_FREQS)
    hf_weights_infant = np.random.uniform(*rng_weights_hf, NUM_HF_FREQS)

    # Generate random phase shifts for each band
    phase_shifts_adult = np.random.uniform(0, 2 * np.pi, NUM_VLF_FREQS + NUM_LF_FREQS + NUM_HF_FREQS)
    phase_shifts_infant = np.random.uniform(0, 2 * np.pi, NUM_VLF_FREQS + NUM_LF_FREQS + NUM_HF_FREQS)

    # Collect parameters in dict
    individual = {}
    individual['base_ibi_adult'] = base_ibi_adult
    individual['base_ibi_infant'] = base_ibi_infant
        
    for i in range(NUM_VLF_FREQS):
        individual[f"vlf_freq_{i}_adult"] = vlf_freqs_adult[i]
        individual[f"vlf_weight_{i}_adult"] = vlf_weights_adult[i]   
        individual[f"vlf_freq_{i}_infant"] = vlf_freqs_infant[i]
        individual[f"vlf_weight_{i}_infant"] = vlf_weights_infant[i]
        individual[f"vlf_phase_{i}_infant"] = phase_shifts_infant[i]
        individual[f"vlf_phase_{i}_adult"] = phase_shifts_adult[i]
    for i in range(NUM_LF_FREQS):
        individual[f"lf_freq_{i}_adult"] = lf_freqs_adult[i]
        individual[f"lf_weight_{i}_adult"] = lf_weights_adult[i]
        individual[f"lf_freq_{i}_infant"] = lf_freqs_infant[i]
        individual[f"lf_weight_{i}_infant"] = lf_weights_infant[i]
        individual[f"lf_phase_{i}_infant"] = phase_shifts_infant[i + NUM_VLF_FREQS]
        individual[f"lf_phase_{i}_adult"] = phase_shifts_adult[i + NUM_VLF_FREQS]
    for i in range(NUM_HF_FREQS):
        individual[f"hf_sfreq_{i}_adult"] = hf_freqs_adult[i]
        individual[f"hf_weight_{i}_adult"] = hf_weights_adult[i]
        individual[f"hf_freq_{i}_infant"] = hf_freqs_infant[i]
        individual[f"hf_weight_{i}_infant"] = hf_weights_infant[i]
        individual[f"hf_phase_{i}_infant"] = phase_shifts_infant[i + NUM_VLF_FREQS + NUM_LF_FREQS]
        individual[f"hf_phase_{i}_adult"] = phase_shifts_adult[i + NUM_VLF_FREQS + NUM_LF_FREQS]


    return individual

def initialize_population(population_size: int):
    '''
    TODO: documentation
    '''
    population = [initialize_individual() for _ in range(population_size)]
    return np.array(population)

def evaluate_fitness_individual(individual: dict, target_zlc: float, distance_metric: str='euclidian'):
    '''
    Calculates the fitness of an individual based on the deviation of its measured zero-lag coefficient (ZLC) of RSA cross-correlation from a target ZLC.

    This function is used within a genetic algorithm to assess the fitness of each individual. 
    It evaluates how well the individual's parameters generate Interbeat Interval (IBI) patterns that align with a desired level of RSA synchrony, as measured by the zero-lag coefficient (ZLC) of RSA cross-correlation. 
    Fitness is quantified as the deviation between the calculated ZLC and the target ZLC, using a specified distance metric.

    Parameters:
    - individual (dict): A dictionary containing key-value pairs for the 98 parameters of a dyad IBI generator. The specifics of these parameters are detailed in the README.
    - target_zlc (float): The target zero-lag coefficient that represents the desired level of RSA synchrony.
    - distance_metric (str, optional): The metric used for calculating the deviation between the measured and target ZLC. Options include 'euclidian' and 'log'. Defaults to 'euclidian'.

    Returns:
    float: The calculated fitness value. Lower values indicate a closer match to the target ZLC. Returns float('inf') in case of an exception during calculation.
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
        abs_distance = abs(zlc - target_zlc)
        match distance_metric:
            # absolute distance
            case 'euclidian':
                return abs_distance
            # log scaled distance
            case 'log':
                return abs_distance if abs_distance == 0 else np.log(abs_distance)
            # default case (euclidian)
            case _:
                return abs(zlc - target_zlc)
    
    # return infinity on exception
    except ValueError:
        return float('inf')
    
def evaluate_fitness(population: np.array, target_zlc: float, distance_metric: str='euclidian'):
    '''
    Evaluates the fitness of each individual in the population based on the deviation from the ideal zero-lag coefficient (ZLC) of RSA cross-correlation.

    This function assesses the fitness of a population in a genetic algorithm. Each individual's fitness is determined by how closely its zero-lag coefficient (a measure derived from RSA cross-correlation) matches a target value. 
    The deviation between the measured and target ZLC is calculated using a specified distance metric.

    Parameters:
    - population (np.array): An array of individuals in the current generation, where each individual is represented as a dictionary of parameters.
    - target_zlc (float): The target zero-lag coefficient value to aim for. This value is used as a benchmark to assess the fitness of individuals.
    - distance_metric (str, optional): The metric used to calculate the distance (deviation) between the measured ZLC of an individual and the target ZLC. Options are 'euclidian' and 'log'. Defaults to 'euclidian'.

    Returns:
    np.array: An array of fitness values, one for each individual in the population. Lower fitness values indicate a closer match to the target ZLC and thus a more desirable solution.
    '''
    return [evaluate_fitness_individual(individual, target_zlc, distance_metric) for individual in population]
    
def select_parents(population: np.array, fitness: np.array):
    '''
    Selects parents from the population using Stochastic Universal Sampling (SUS).

    This function implements Stochastic Universal Sampling, a selection method used in genetic algorithms to choose parent individuals for the next generation. 
    SUS is designed to give each individual a chance of selection proportional to its fitness. 
    It reduces the chance of premature convergence by preventing the fittest individuals from dominating the selection process too early.

    Parameters:
    - population (np.array): An array of individuals in the current generation. Each individual is a dictionary of gene names and values.
    - fitness (np.array): An array of fitness values corresponding to each individual in the population. Higher fitness values indicate better individuals.

    Returns:
    np.array: An array of selected parent individuals.

    Notes:
    - The number of parents selected is half the size of the input population.
    - Fitness values are normalized to create a probability distribution.
    - The function employs a 'roulette wheel' approach with evenly spaced pointers, determined by the step size and a random start point. This method ensures a spread-out selection across the population's fitness range.
    '''
    # set number of parents to be selected - this can be subject to experimentation (influences competition and performance)
    num_parents = population.shape[0]// 2

    # initialize parent array
    parents = np.empty(num_parents, dtype=object)

    # Normalize fitness values
    total_fitness = np.sum(fitness)
    normalized_fitness = fitness / total_fitness

    # Calculate cumulative sum
    cumulative_sum = np.cumsum(normalized_fitness)
    
    # Determine the step size and the start point
    step = 1.0 / num_parents
    start = np.random.uniform(0, step)
    
    # Select individuals as parents
    idx = 0
    for i in range(num_parents):
        pointer = start + i * step
        while cumulative_sum[idx] < pointer:
            idx += 1
        parents[i] = population[idx]

    return parents

def crossover_arithmetic(parent0_v: np.array, parent1_v: np.array, alpha:float = .3):
    '''
    Performs arithmetic crossover between two parent individuals to generate two offspring.

    This function employs arithmetic crossover, a technique that combines the genetic material (parameter vectors) of two parents to produce offspring. 
    Each gene in the offspring is a weighted average of the corresponding genes in the parents, with the weighting factor being 'alpha'. 
    This method ensures that the offspring's genes are within the range defined by their parents' genes, promoting a balanced exploration of the solution space.

    Parameters:
    - parent0_v (np.array): The parameter vector of the first parent.
    - parent1_v (np.array): The parameter vector of the second parent.
    - alpha (float, optional): The weighting factor used in the arithmetic combination of parent genes. Defaults to 0.3. The value determines the bias towards one parent's genes over the other in the offspring.

    Returns:
    Tuple[np.array, np.array]: Two numpy arrays representing the parameter vectors of the two generated offspring.

    Notes:
    - The function calculates the offspring's genes as a weighted average of the corresponding genes from each parent, controlled by the 'alpha' parameter.
    - The choice of 'alpha' affects how closely the offspring resemble their parents. An alpha of 0.5 results in offspring that are an equal mix of both parents, while other values bias the offspring towards one parent.
    - This crossover method is particularly useful for maintaining genetic diversity within a range defined by the parent genes.
    '''
    child0_v = alpha * parent0_v + (1-alpha) * parent1_v
    child1_v = alpha * parent1_v + (1-alpha) * parent0_v
    return child0_v, child1_v

def crossover_blend(parent0_v: np.array, parent1_v: np.array, alpha: float=.5):
    '''
    Performs blend crossover (BLX-alpha) between two parent individuals to generate two offspring.

    This function takes the genetic material (parameter vectors) of two parents and applies blend crossover, a technique that creates offspring whose genes are random values chosen from a range defined by the parents' genes and an expansion factor 'alpha'. 
    This method allows for the offspring to potentially have genes that are outside the range of the parent genes, promoting diversity in the population.

    Parameters:
    - parent0_v (np.array): The parameter vector of the first parent.
    - parent1_v (np.array): The parameter vector of the second parent.
    - alpha (float, optional): The expansion factor used in determining the range for gene values in the offspring. Defaults to 0.5. Higher values increase the range and hence the potential diversity of the offspring. Must be in range [0, 1].

    Returns:
    Tuple[np.array, np.array]: Two numpy arrays representing the parameter vectors of the two generated offspring.

    Notes:
    - The function calculates the gene range as the absolute difference between corresponding genes of the parents.
    - It then expands this range by 'alpha' on both sides (lower and upper) to create a new range for each gene.
    - Offspring genes are randomly selected within these expanded ranges, allowing for traits that may be beyond the parents' traits.
    '''
    # Calculate the range for each gene
    gene_range = np.abs(parent0_v - parent1_v)
    min_gene = np.minimum(parent0_v, parent1_v) - alpha * gene_range
    max_gene = np.maximum(parent0_v, parent1_v) + alpha * gene_range

    # Generate offspring within the range
    child0_v = np.random.uniform(min_gene, max_gene)
    child1_v = np.random.uniform(min_gene, max_gene)

    return child0_v, child1_v

def crossover(parents: np.array, crossover_method: str):
    '''
    Performs crossover operation on a set of parent individuals to generate offspring.

    This function takes an array of parent individuals and applies a crossover method to generate offspring. 
    The crossover methods available are 'arithmetic' and 'blend' (also referred to as blx-alpha). 
    Each pair of parents generates two offspring, with genes combined according to the specified crossover method. 
    The function ensures that the resulting offspring conform to the range limits of their parameters.

    Parameters:
    - parents (np.array): An array of parent individuals, where each individual is a dictionary of gene names and values.
    - crossover_method (str): The method of crossover to be used. Options include 'arithmetic' and 'blend'.

    Returns:
    np.array: An array of offspring individuals, each represented as a dictionary of gene names and values.

    Notes:
    - The offspring array size is adjusted to be even, as each pair of parents produces two offspring.
    - The crossover operation is applied to the parameter vectors of the parents, and the results are converted back into dictionaries for the offspring.
    '''
    # initialize offspring
    offspring_size = parents.shape[0] - parents.shape[0] % 2
    offspring = np.empty(offspring_size, dtype=object)

    # perform crossover operation
    for i in range(offspring_size):
        # select pair of parents
        idx0 = i*2
        idx1 = i*2 + 1
        parent0 = parents[idx0]
        parent1 = parents[idx1]

        # extract parameter vector from parents
        parent0_v = np.array(list(parent0.values()))
        parent1_v = np.array(list(parent1.values()))

        # extract parameter names
        param_names = parent0.keys()

        # create children as parameter voctors by crossover method
        match crossover_method:
            case 'arithmetic':
                child0_v, child1_v = crossover_arithmetic(parent0_v, parent1_v)
            case 'blend':
                child0_v, child1_v = crossover_blend(parent0_v, parent1_v)
            case _:
                child0_v, child1_v = crossover_arithmetic(parent0_v, parent1_v)

        # turn parameter vectors to dicts
        child0 = dict(zip(param_names, child0_v.tolist()))
        child1 = dict(zip(param_names, child1_v.tolist()))
        
        # apply range limits to children and add them to offsprint
        offspring[idx0] = apply_limits(child0)
        offspring[idx1] = apply_limits(child1)
    
    return offspring

def gaussian_mutation(individual, mutation_rate, mutation_scale=0.1):
    '''
    Applies Gaussian mutation to an individual based on a given mutation rate and scale.

    This function iteratively goes through each gene (represented by key-value pairs) in an individual's genetic makeup. 
    For each gene, there is a probability (defined by the mutation rate) that a Gaussian mutation will be applied. 
    The mutation scale, along with the range of possible values for a gene (obtained from `get_limits`), determines the magnitude of the mutation. 

    Parameters:
    - individual (dict): A dictionary representing an individual's genes.
    - mutation_rate (float): The probability of mutating each gene. Must be in the range [0, 1].
    - mutation_scale (float, optional): The scale of the Gaussian distribution used for mutation. Defaults to 0.1. It adjusts the magnitude of changes applied to mutated genes.

    Returns:
    dict: A dictionary representing the mutated individual. The structure is the same as the input individual, but with mutations applied to some of the genes.
    '''
    individual_mutated = {}
    for key, value in individual.items():
        if np.random.rand() < mutation_rate:
            limits = get_limits(key)
            rng = 2*np.pi if 'phase' in key else abs(limits[1] - limits[0])
            individual_mutated[key] = value * np.random.normal(0, mutation_scale * rng)
        else:
            individual_mutated[key] = value
    return individual_mutated

def mutate(offspring: np.array, mutation_rate: float, mutation_scale: float):
    '''
    Applies Gaussian mutation to each individual in the offspring.

    This function iterates over an array of individuals (offspring) and applies Gaussian mutation to each individual based on a specified mutation rate and scale. 
    The mutation introduces variability in the population, which is essential for the genetic algorithm's exploration of the solution space.

    Parameters:
    - offspring (np.array): An array of individuals to be mutated. Each individual is expected to be a numeric array representing its genetic makeup.
    - mutation_rate (float): The probability of each gene being mutated. This value should be in the range [0, 1]. A higher rate increases the likelihood of mutations in the individuals.
    - mutation_scale (float): The scale of randomness applied during mutation, determining the magnitude of the mutation. This value should be in the range [0, 1]. A higher scale results in more significant changes to the genes during mutation.

    Returns:
    np.array: An array of mutated individuals. The structure of the array is similar to the input offspring array, but with mutations applied to the individuals.
    '''
    return np.array([gaussian_mutation(individual, mutation_rate, mutation_scale) for individual in offspring])

def succession(population: np.array, fitness: np.array, crossover_method: str, mutation_rate: float, mutation_scale: float):
    '''
    Generates a new generation of population through the processes of selection, crossover, and mutation.

    This function represents a core step in the genetic algorithm, where a new population is created from the current one. 
    It starts by selecting the fittest individuals as parents. These parents undergo crossover and mutation to produce offspring. The offspring then form the new generation.

    Parameters:
    - population (np.array): The current population array, where each element represents an individual.
    - fitness (np.array): An array of fitness values corresponding to each individual in the population.
    - crossover_method (str): The crossover method to be used for generating offspring.
    - mutation_rate (float): The probability of mutation occurring in an offspring.
    - mutation_scale (float): The scale of mutation when it occurs.

    Returns:
    np.array: The new population formed by concatenating the parents and the mutated offspring.

    Notes:
    - The population and fitness arrays must be of the same length, each entry in the fitness array corresponding to an individual in the population array.
    '''
    # Select the best parents for mating
    parents = select_parents(population, fitness)

    # Generate the next generation using crossover and introcuce some variation through mutation
    offspring = mutate(crossover(parents, crossover_method), mutation_rate, mutation_scale)

    # Create the new population
    new_population = np.concatenate(parents, offspring)
    
    return new_population

def evolution(population_size: int, max_num_generations: int, fitness_thresh: float, target_zlc: float, distance_metric: str, crossover_method: str, mutation_rate: float, mutation_scale: float):
    '''
    Conducts the genetic algorithm's evolution process. 
    The goal is to find parameters for an IBI generation algorithm in order to minimize the distance of the zero-lag coefficient (zlcs) in an RSA Synchrony algorithm to a target zlc.

    This function represents the main optimization loop in a genetic algorithm. 
    It initializes a population and iteratively evolves it over a specified number of generations or until a fitness threshold is met. 
    Each generation involves the creation of a new population through crossover and mutation. 
    The fitness of each individual in the population is evaluated, and the process repeats. 
    The function returns the final population and their respective fitness scores.

    Parameters:
    - population_size (int): The number of individuals in the population.
    - max_num_generations (int): The maximum number of generations to run the evolution for.
    - fitness_thresh (float): The fitness threshold for terminating the evolution early. Evolution stops if any individual's fitness is less than this threshold.
    - target_zlc (float): A target value for the zero-lag coefficient fitness evaluation function.
    - distance_metric (str): The type of distance metric to be used in the fitness evaluation. [options: 'euclidian', 'log']
    - crossover_method (str): The crossover method to be used for generating new individuals. [options: 'arithmetic', 'blend']
    - mutation_rate (float): The probability of mutation occurring in an individual.
    - mutation_scale (float): The scale of mutation when it occurs.

    Returns:
    Tuple[List, List]: A tuple containing two elements:
        - The final population after the evolution process.
        - The fitness scores of the final population.

    Note:
    Logging can be toggled through the global variable LOG at the top of this script.
    '''
    # initialize population and fitness
    population = initialize_population(population_size)
    fitness = evaluate_fitness(population, target_zlc, distance_metric)

    # iterate over generations
    for generation_index in range(max_num_generations):

        # create new generation of population
        population = succession(population, fitness, crossover_method, mutation_rate, mutation_scale)

        # calculate fitness
        fitness = evaluate_fitness(population, target_zlc, distance_metric)
        best_fitness = np.min(fitness)        

        # inform about fitness state
        if LOG:
            print(f"# Generation {generation_index} - best fitness: {best_fitness}")

        # terminate evolution if desired fitness is reached
        if best_fitness < fitness_thresh:
            break

    return population, fitness