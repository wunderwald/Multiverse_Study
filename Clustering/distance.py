import numpy as np

'''
MORE METRICS

- generate IBI sequence
    -> then use standard deviation of IBI seq

- combined metrics: stamdard deviation, weight distribution, ibi similariry and more, combine it to one distance value



'''



def distance_base_ibi(a: dict, b: dict):
    '''
    Calculate distance between two data points based on distribution of weights over frequency bands (vlf, lf, hf).

    Params:
    - a (dict): first dyad ibi parameters and optimization hyperparameters
    - b (dict): second dyad ibi parameters and optimization hyperparameters

    Returns: Distance value (float)
    '''
    base_ibi_adult_a = a['dyad_parameters']['base_ibi_adult']
    base_ibi_adult_b = b['dyad_parameters']['base_ibi_adult']
    base_ibi_infant_a = a['dyad_parameters']['base_ibi_infant']
    base_ibi_infant_b = b['dyad_parameters']['base_ibi_infant']

    return abs(base_ibi_adult_a - base_ibi_adult_b) + abs(base_ibi_infant_a - base_ibi_infant_b)

def distance_band(a: dict, b: dict, band: str):
    '''
    Calculate distance between two data points based on frequency parameters (freq, weight, phase) in a selected frequency band.

    Params:
    - a (dict): first dyad ibi parameters and optimization hyperparameters
    - b (dict): second dyad ibi parameters and optimization hyperparameters
    - band (string): id of the selected custom distance metric. Options: 'vlf', 'lf', 'hf'

    Returns: Distance value (float)
    '''
    return 'TODO'

def distance_band_vlf(a: dict, b: dict):
    '''
    Helper function for distance_band(), sets band='vlf'
    '''
    return distance_band(a=a, b=b, band='vlf')

def distance_band_lf(a: dict, b: dict):
    '''
    Helper function for distance_band(), sets band='lf'
    '''
    return distance_band(a=a, b=b, band='lf')

def distance_band_hf(a: dict, b: dict):
    '''
    Helper function for distance_band(), sets band='hf'
    '''
    return distance_band(a=a, b=b, band='hf')

def distance_weight_distribution(a: dict, b: dict):
    '''
    Calculate distance between two data points based on distribution of weights over frequency bands (vlf, lf, hf).

    Params:
    - a (dict): first dyad ibi parameters and optimization hyperparameters
    - b (dict): second dyad ibi parameters and optimization hyperparameters

    Returns: Distance value (float)
    '''
    return 'TODO'

def get_distance_metric(method: str):
    '''
    Returns a distance metric function based on the selected metric.

    Params:
    - method (string): id of the selected custom distance metric. 

    Returns: distance metric function (callable)
    '''
    if method == 'weight_distribution':
        return distance_weight_distribution
    if method == 'band_vlf':
        return distance_band_vlf
    if method == 'band_lf':
        return distance_band_lf
    if method == 'band_hf':
        return distance_band_hf
    if method == 'base_ibi':
        return distance_base_ibi
    return lambda a, b: print('! invalid metric')

def create_distance_matrix(data, distance_metric):
    n = len(data)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            distance_matrix[i, j] = distance_metric(data[i], data[j])
            distance_matrix[j, i] = distance_matrix[i, j]  # Distance matrix is symmetric
    return distance_matrix