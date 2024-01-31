import ibi_generator as ibi
import rsa_drew as rsa
import numpy as np

# define constants for IBI generation
RECORDING_TIME = 300
NUM_VLF_BANDS = 4
NUM_LF_BANDS = 6
NUM_HF_BANDS = 6

# define parameter ranges for IBI generation
PARAM_RANGES = {
    'freq_vlf': [0.01, 0.04],
    'freq_lf': [0.04, 0.15],
    'freq_hf': [0.15, 0.4],
    'weights_vlf': [0.015, 1.0],
    'weights_lf': [0.004, 0.4],
    'weights_hf': [0.002, 0.2],
    'phase_shift': [0, 2 * np.pi],
    'base_ibi_adult': [650, 750],
    'base_ibi_infant': [450, 550]
}

# define constants for parameter optimization
TARGET_ZLC = 300

# define hyper-parameter ranges for parameter optimization
HYPERPARAM_RANGES = {}


# define objective function
def obj_rsa_sync_zlc(params: dict, target_zlc: float):
    '''
    Objective function for calculating RSA synchrony using Drew's algorithm. 
    Synchrony is measured using the zero-lag coefficient (ZLC).
    The error is calculated as the absolute difference between the ZLC and a target ZLC.

    Parameters:
    - params (dict): key-value pairs for the 98 parameters for dyad IBI generator (see README for details)
    - target_zlc (float): zero lag coefficient value that is the optimization target 

    Returns:
    - err (float): the absolute difference between calculated and target ZLC, float('inf') on error
    '''

    # validate parameters
    # TODO test if parameters are in range, return float('inf') on fail

    # extract parameters
    adult_params = params['adult']
    infant_params = params['adult']

    try:
        # generate IBIs
        adult_ibi, infant_ibi = ibi.generate_dyad_ibi(RECORDING_TIME, adult_params, infant_params)

        # calculate synchrony
        zlc, _ = rsa.rsa_synchrony(adult_ibi, infant_ibi)

        # calculate error
        err = abs(zlc - target_zlc)

        return err
    
    # return infinity on error
    except ValueError:
        return float('inf')