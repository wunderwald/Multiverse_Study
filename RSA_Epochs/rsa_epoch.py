import neurokit2 as nk2
import numpy as np
import scipy.stats

# --------------------------------
# HELPERS
def split_ibi_epochs(ibi_ms, epoch_length_ms):
    epochs = []
    current_epoch = []
    ibi_sum = 0
    for ibi_sample in ibi_ms:
        new_sum = ibi_sum + ibi_sample
        print(ibi_sample, new_sum, epoch_length_ms)
        if new_sum > epoch_length_ms:
            # current epoch is done
            epochs.append(current_epoch)
            ibi_sum = ibi_sample
            current_epoch = [ibi_sample]
            continue
        # append to current epoch
        current_epoch.append(ibi_sample)
        ibi_sum = new_sum
    epochs.append(current_epoch)
    return epochs
# HELPERS END
# --------------------------------

def synchrony(epochs_a, epochs_b, sync_type):
    '''
    Calculates synchrony based on a type of synchrony measure.

    Parameters:
    - epochs_a: (array-like): list of rsa epoch values.
    - epochs_b: (array-like): list of rsa epoch values.
    - sync_type (str): Options are 'pearson_corr', 'spearman_corr', 'kendall_corr',

    Returns:
    - synchrony_score (float): the synchrony score
    '''
    match sync_type:
        case 'pearson_corr':
            return scipy.stats.pearsonr(epochs_a, epochs_b)[0]
        case 'spearman_corr':
            return scipy.stats.spearmanr(epochs_a, epochs_b)[0]
        case 'kendall_corr':
            return scipy.stats.kendalltau(epochs_a, epochs_b)[0]
        case _:
            return float('nan')

def rsa_epochs(ibi_ms, epoch_length_ms, rsa_method):
    '''
    Calculates epoch-based RSA in epochs of a set length.

    RSA can be estimated using different techniques:
    - hf-hrv: high-frequency heart-rate variability is interpreted as RSA
    - drew: Drew Abbneys RSA algorithm is calcuated for consecutive, non-overlapping windows (aka epochs) 

    Parameters:
    - ibi_ms (array-like): list of ibi samples in ms
    - epoch_length_ms (float): (max) length of epochs in ms
    - rsa_method (string): options are 'hf-hrv', 'drew'...

    Returns:
    - epochs (array): list of epoch-based rsa values 

    '''

    # split ibi samples into epochs
    pass

def rsa_epoch_synchrony(ibi_a_ms, ibi_b_ms, epoch_length_ms=3000, rsa_method='hf-hrv', sync_type='pearson_corr'):
    '''
    Calculates epoch based rsa for a pair of ibi sequences. Using a given synchrony measure, a synchrony score is calculated for the pair of rsa epochs.

    Parameters:
    - sync_type (string): see options in synchrony()

    Returns:
    - synchrony_score (float): the synchrony score
    '''
    epochs_a = rsa_epochs(ibi_a_ms, epoch_length_ms, rsa_method)
    epochs_b = rsa_epochs(ibi_b_ms, epoch_length_ms, rsa_method)
    synchrony_score = synchrony(epochs_a, epochs_b, sync_type)
    return synchrony_score


