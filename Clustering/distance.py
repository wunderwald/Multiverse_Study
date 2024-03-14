def distance_weight_distribution(a: dict, b: dict):
    '''
    Calculate distance between two data points based on distribution of weights over frequency bands (vlf, lf, hf).

    Params:
    - a (dict): first dyad ibi parameters and optimization hyperparameters
    - b (dict): second dyad ibi parameters and optimization hyperparameters

    Returns: Distance value (float)
    '''
    return 'TODO'

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

def distance_base_ibi(a: dict, b: dict):
    '''
    Calculate distance between two data points based on distribution of weights over frequency bands (vlf, lf, hf).

    Params:
    - a (dict): first dyad ibi parameters and optimization hyperparameters
    - b (dict): second dyad ibi parameters and optimization hyperparameters

    Returns: Distance value (float)
    '''
    return 'TODO'

def distance(a: dict, b: dict, method: str):
    '''
    Calculate distance between two data points based on a custom distance metric.

    Params:
    - a (dict): first dyad ibi parameters and optimization hyperparameters
    - b (dict): second dyad ibi parameters and optimization hyperparameters
    - method (string): id of the selected custom distance metric. Options: 'weight_distribution', 'band_vlf', 'band_lf', 'band_hf', 'base_ibi'

    Returns: distance value (float), or infinity if method id is invalid
    '''
    if method is 'weight_distribution':
        return distance_weight_distribution(a=a, b=b)
    if method is 'band_vlf':
        return distance_band(a=a, b=b, band='vlf')
    if method is 'band_lf':
        return distance_band(a=a, b=b, band='lf')
    if method is 'band_hf':
        return distance_band(a=a, b=b, band='hf')
    if method is 'base_ibi':
        return distance_base_ibi(a=a, b=b)
    return float('inf')