import math
import scipy.stats
import neurokit2 as nk2
from neurokit2.hrv.hrv_rsa import _hrv_rsa_pb
from resample import resample_ibi
from rsa_drew import rsa_magnitude_adult, rsa_magnitude_infant

# --------------------------------
# HELPERS
def split_ibi_epochs(ibi_ms, epoch_length_ms):
    epochs = []
    current_epoch = []
    ibi_sum = 0
    for ibi_sample in ibi_ms:
        new_sum = ibi_sum + ibi_sample
        if new_sum > epoch_length_ms:
            # current epoch is done
            epochs.append(current_epoch)
            ibi_sum = ibi_sample
            current_epoch = [ibi_sample]
            continue
        # append to current epoch
        current_epoch.append(ibi_sample)
        ibi_sum = new_sum
    if(len(current_epoch) > 0):
        epochs.append(current_epoch)
    return epochs
def ibi_to_peaks(ibi_ms):
    peaks = [0]
    ibi_sum = 0
    for ibi_sample in ibi_ms:
        ibi_sum += ibi_sample
        peaks.append(ibi_sum)
    return peaks

# HELPERS END
# --------------------------------

def synchrony(rsa_epochs_a, rsa_epochs_b, sync_type):
    '''
    Calculates synchrony based on a type of synchrony measure.

    Parameters:
    - rsa_epochs_a: (array-like): list of rsa epoch values.
    - rsa_epochs_b: (array-like): list of rsa epoch values.
    - sync_type (str): Options are 'pearson_corr', 'spearman_corr', 'kendall_corr',

    Returns:
    - synchrony_score (float): the synchrony score
    '''

    # match length of epochs
    num_epochs_a = len(rsa_epochs_a)
    num_epochs_b = len(rsa_epochs_b)
    if(num_epochs_a != num_epochs_b):
        min_num_epochs = num_epochs_a if num_epochs_a < num_epochs_b else num_epochs_b
        del rsa_epochs_a[min_num_epochs:]
        del rsa_epochs_b[min_num_epochs:]

    # remove trailing NaN (due to shorter last epoch)
    while(math.isnan(rsa_epochs_a[-1]) or math.isnan(rsa_epochs_b[-1])):
        del rsa_epochs_a[-1]
        del rsa_epochs_b[-1]

    # calculate synchrony
    match sync_type:
        case 'pearson_corr':
            return scipy.stats.pearsonr(rsa_epochs_a, rsa_epochs_b)[0]
        case 'spearman_corr':
            return scipy.stats.spearmanr(rsa_epochs_a, rsa_epochs_b)[0]
        case 'kendall_corr':
            return scipy.stats.kendalltau(rsa_epochs_a, rsa_epochs_b)[0]
        case _:
            return float('nan')

def rsa_hf_hrv(ibi_ms):
    peaks = ibi_to_peaks(ibi_ms)
    hrv = nk2.hrv_frequency(peaks, sampling_rate=1000)
    hrv_hf_power = hrv['HRV_HF'][0]
    return hrv_hf_power

def rsa_porges_bohrer(ibi_ms):
    resampling_rate = 100
    ibi_resampled = resample_ibi(ibi_ms, resampling_rate)
    rsa = _hrv_rsa_pb(ibi_resampled, resampling_rate)
    return rsa["RSA_PorgesBohrer"]

def rsa_per_epoch(ibi_ms, epoch_length_ms, rsa_method, age_type='adult'):
    '''
    Calculates epoch-based RSA in epochs of a set length.

    RSA can be estimated using different techniques:
    - high-frequency heart-rate variability is interpreted as RSA
    - porges_bohrer: porges bohrer algorithm implementation in neurokit2
    - drew: Drew Abbneys RSA algorithm is calcuated for consecutive, non-overlapping windows (variant/implementation of porges bohrer)

    Parameters:
    - ibi_ms (array-like): list of ibi samples in ms
    - epoch_length_ms (float): (max) length of epochs in ms
    - rsa_method (string): options are 'hf_hrv', 'drew', 'porges_bohrer'...
    - age_type (string): 'adult' (default) or 'infant (only needed for drew RSA)

    Returns:
    - epochs (array): list of epoch-based rsa values 

    '''

    # split ibi samples into epochs
    epochs = split_ibi_epochs(ibi_ms, epoch_length_ms)

    # remove inclomplete last epoch
    del epochs[-1]

    # calculate RSA for each epoch
    rsa_epochs = []
    for epoch in epochs:
        rsa_value = float('nan')
        match rsa_method:
            case 'hf_hrv':
                rsa_value = rsa_hf_hrv(epoch)
            case 'porges_bohrer':
                rsa_value = rsa_porges_bohrer(epoch)
            case 'drew':
                rsa_value = rsa_magnitude_infant(epoch) if age_type == 'infant' else rsa_magnitude_adult(epoch)
        rsa_epochs.append(rsa_value)
    return rsa_epochs

def rsa_epoch_synchrony(ibi_a_ms, ibi_b_ms, epoch_length_ms=30000, rsa_method='hf_hrv', sync_type='pearson_corr', dyad_type='adult_infant'):
    '''
    Calculates epoch based rsa for a pair of ibi sequences. Using a given synchrony measure, a synchrony score is calculated for the pair of rsa epochs.

    Parameters:
    - ibi_a_ms (array-like): ibi signal a in milliseconds
    - ibi_b_ms (array-like): ibi signal b in milliseconds
    - epoch_length_ms (int): epoch length in milliseconds
    - rsa_method (string): method of calculating rsa magnitude. Options: 'drew', 'hf_hrv', 'porges_bohrer'
    - sync_type (string): see options in synchrony()
    - dyad_type (string): 'adult-infant' (default) or 'adult-adult' dyad (only important for drew RSA).

    Returns:
    - synchrony_score (float): the synchrony score
    '''
    # determine age types (only needed for drew RSA)
    age_type_a = 'adult'
    age_type_b = 'adult' if dyad_type == 'adult_adult' else 'infant'

    # calculate rsa epochs
    epochs_a = rsa_per_epoch(ibi_a_ms, epoch_length_ms, rsa_method, age_type_a)
    epochs_b = rsa_per_epoch(ibi_b_ms, epoch_length_ms, rsa_method, age_type_b)

    # calculeate synchrony between the two rsa epoch lists
    synchrony_score = synchrony(epochs_a, epochs_b, sync_type)
    return synchrony_score


