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
def rsa_sync_zlc(params, target_zlc):

    # extract parameters
    adult_params = params['adult']
    infant_params = params['adult']

    # generate IBIs
    adult_ibi, infant_ibi = ibi.generate_dyad_ibi(RECORDING_TIME, adult_params, infant_params)

    # calculate synchrony
    zlc, _ = rsa.rsa_synchrony(adult_ibi, infant_ibi)

    # calculate error
    err = abs(zlc - target_zlc)

    return err