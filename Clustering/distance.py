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

def distance_base_ibi(a: dict, b: dict):
    '''
    Calculate distance between two data points based on distribution of weights over frequency bands (vlf, lf, hf).

    Params:
    - a (dict): first dyad ibi parameters and optimization hyperparameters
    - b (dict): second dyad ibi parameters and optimization hyperparameters

    Returns: Distance value (float)
    '''
    return 'TODO'