import neurokit2 as nk2
import numpy as np
import scipy.stats
import math

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
    hrv_power = nk2.hrv_frequency(peaks, sampling_rate=1000)
    hrv_hf_power = hrv_power['HRV_HF'][0]
    return hrv_hf_power

def rsa_drew_single_window(ibi_ms):
    pass

def rsa_per_epoch(ibi_ms, epoch_length_ms, rsa_method):
    '''
    Calculates epoch-based RSA in epochs of a set length.

    RSA can be estimated using different techniques:
    - hf_hrv: high-frequency heart-rate variability is interpreted as RSA
    - drew: Drew Abbneys RSA algorithm is calcuated for consecutive, non-overlapping windows (aka epochs) 

    Parameters:
    - ibi_ms (array-like): list of ibi samples in ms
    - epoch_length_ms (float): (max) length of epochs in ms
    - rsa_method (string): options are 'hf_hrv', 'drew'...

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
            case 'drew':
                rsa_value = rsa_drew_single_window(epoch)
        rsa_epochs.append(rsa_value)
    return rsa_epochs

def rsa_epoch_synchrony(ibi_a_ms, ibi_b_ms, epoch_length_ms=30000, rsa_method='hf_hrv', sync_type='pearson_corr'):
    '''
    Calculates epoch based rsa for a pair of ibi sequences. Using a given synchrony measure, a synchrony score is calculated for the pair of rsa epochs.

    Parameters:
    - sync_type (string): see options in synchrony()

    Returns:
    - synchrony_score (float): the synchrony score
    '''
    epochs_a = rsa_per_epoch(ibi_a_ms, epoch_length_ms, rsa_method)
    epochs_b = rsa_per_epoch(ibi_b_ms, epoch_length_ms, rsa_method)
    synchrony_score = synchrony(epochs_a, epochs_b, sync_type)
    return synchrony_score


