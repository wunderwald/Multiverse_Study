import ibi_generator as ibi
import rsa_drew as rsa

# define parameter ranges for IBI generation
PARAM_RANGES = {
    'freq_vlf': [],
    'freq_lf': [],
    'freq_hf': [],
    'weights_vlf': [],
    'weights_lf': [],
    'weights_hf': [],
    'phase_shift': [],
    'base_ibi_adult': [],
    'base_ibi_infant': []
}

# define hyper-parameter ranges for parameter optimization
HYPERPARAM_RANGES = {}

# define constants for IBI generation
RECORDING_TIME = 300

# define constants for parameter optimization
TARGET_ZLC = 300

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