import random

OPTIONS_CROSSOVER_METHOD = {
    'shuffle': 0.6,
    'arithmetic': 0.25,
    'blend': 0.15,
}
OPTIONS_SELECT_PARENTS_METHOD = {
    'sus': .85,
    'roulette_fittest': .15
}
RANGE_MUTATION_RATE = [.05, .3]
RANGE_MUTATION_SCALE = [.55, .95]
RANGE_PARENT_RATIO = [.05, .45]
RANGE_POPULATION_SIZE = [80, 180]

def pick_option(weighted_options):
    options = list(weighted_options.keys())
    probabilities = list(weighted_options.values())
    return random.choices(options, weights=probabilities, k=1)[0]

def pick_from_range(rng):
    min, max = rng
    return random.uniform(min, max)

def random_hyperparams():
    return {
        'CROSSOVER_METHOD': pick_option(OPTIONS_CROSSOVER_METHOD),
        'SELECT_PARENTS_METHOD': pick_option(OPTIONS_SELECT_PARENTS_METHOD),
        'MUTATION_RATE': pick_from_range(RANGE_MUTATION_RATE),
        'MUTATION_SCALE': pick_from_range(RANGE_MUTATION_SCALE),
        'PARENT_RATIO': pick_from_range(RANGE_PARENT_RATIO),
        'POPULATION_SIZE': int(pick_from_range(RANGE_POPULATION_SIZE))
    }

def default_hyperparams():
    return {
        'CROSSOVER_METHOD': 'shuffle',
        'SELECT_PARENTS_METHOD': 'sus',
        'MUTATION_RATE': .1,
        'MUTATION_SCALE': .9,
        'PARENT_RATIO': 0.1,
        'POPULATION_SIZE': 120
    }
    