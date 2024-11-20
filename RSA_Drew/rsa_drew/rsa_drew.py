import pkg_resources
import numpy as np
import pandas as pd
import os
import time
from scipy.signal import convolve, detrend
from scipy.interpolate import interp1d
from .resampled_ibi_ts import resampled_IBI_ts
from .poly_filter_data_2011 import poly_filter_data_2011
from .sliding_window import sliding_window_log_var

def rsa_adult(adult_ibi):
    """
    This function calculates continuous RSA for an adult using a Porges-Bohrer approach.

    Parameters:
    - adult_ibi (array-like): list of adult IBIs in ms
    
    Returns:
    - RSA_A_filt (array-like): a filtered, continuous RSA signal
    - r_A (array-like): the input RSA data, resampled to 5hz
    """

    # load filter
    filt_A_path = pkg_resources.resource_filename('rsa_drew', 'adult_rsa_5Hz_cLSq.csv')
    filt_A = pd.read_csv(filt_A_path).to_numpy().flatten()

    # resample IBI data to 5Hz
    r_A = resampled_IBI_ts(adult_ibi, 5, False)

    # get RSA/BPM and filter RSA
    RSA_A, BPM_A = poly_filter_data_2011(r_A[:, 1], 51, True) 
    RSA_A_filt = convolve(RSA_A, filt_A, mode='valid')

    return RSA_A_filt, r_A
   

def rsa_infant(infant_ibi):
    """
    This function calculates continuous RSA for an infant using a Porges-Bohrer approach.

    Parameters:
    - adult_ibi (array-like): list of infant IBIs in ms
    
    Returns:
    - RSA_I_filt (array-like): a filtered, continuous RSA signal
    - r_I (array-like): the input RSA data, resampled to 5hz
    """
    # load filter
    filt_I_path = pkg_resources.resource_filename('rsa_drew', 'child_RSA.csv')
    filt_I = pd.read_csv(filt_I_path).to_numpy().flatten()

    # resample IBI data to 5Hz
    r_I = resampled_IBI_ts(infant_ibi, 5, False)

    # get RSA/BPM and filter RSA
    RSA_I, BPM_I = poly_filter_data_2011(r_I[:, 1], 51, True)
    RSA_I_filt = convolve(RSA_I, filt_I, mode='valid')   

    return RSA_I_filt, r_I

def rsa_magnitude_adult(adult_ibi):
    """
    This function calculates (single-value) RSA magnitude for an adult using a Porges-Bohrer approach.

    Parameters:
    - adult_ibi (array-like): list of adult IBIs in ms
    
    Returns:
    - rsa_magnitude (float): RSA magnitude measured as the log of the variance of a filtered, continuous RSA signal
    """
    return np.log(np.var(rsa_adult(adult_ibi)))

def rsa_magnitude_infant(infant_ibi):
    """
    This function calculates (single-value) RSA magnitude for an infant using a Porges-Bohrer approach.

    Parameters:
    - infant_ibi (array-like): list of infant IBIs in ms
    
    Returns:
    - rsa_magnitude (float): RSA magnitude measured as the log of the variance of a filtered, continuous RSA signal
    """
    return np.log(np.var(rsa_infant(infant_ibi)))

def rsa_synchrony(mother_ibi, infant_ibi):
    """
    This function calculates the zero lag coefficient and the full cross-correlation function (ccf)
    between detrended, logarithmic variances of filtered Respiratory Sinus Arrhythmia (RSA) signals 
    from a mother-infant dyad.

    Parameters:
    - mother_ibi (array-like): list of mother's IBIs in ms
    - infant_ibi (array-like): list of infant's IBIs in ms
    
    Returns:
    - zeroLagCoefficient (float): The zero lag coefficient in the cross-correlation function, indicating the degree of synchrony at zero time lag between the RSA signals.
    - ccf (numpy.ndarray): The full cross-correlation function (limited by maxlag) between the detrended RSA signals of the mother and infant.

    Raises:
    - ValueError: If the length of filtered RSA data is insufficient for further processing.
    """

    # calculate RSA for mother and infant
    RSA_M_filt, r_M = rsa_adult(mother_ibi)
    RSA_I_filt, _ = rsa_infant(infant_ibi)

    # interpolate filtered RSA data
    if len(RSA_M_filt) < 2 or len(RSA_I_filt) < 2:
        raise ValueError("! Insufficient length of filtered RSA data")

    f = interp1d(np.arange(len(RSA_M_filt)), RSA_M_filt)
    RSA_M_filt_intpl = f(np.linspace(0, len(RSA_M_filt) - 1, len(r_M)))
    f = interp1d(np.arange(len(RSA_I_filt)), RSA_I_filt)
    RSA_I_filt_intpl = f(np.linspace(0, len(RSA_I_filt) - 1, len(r_M)))

    # calculate log of variance with sliding window
    window_size = 74  # 15 seconds window at 5 Hz sampling rate
    lv_RSA_M_fif_raw = sliding_window_log_var(RSA_M_filt_intpl, window_size)
    lv_RSA_I_fif_raw = sliding_window_log_var(RSA_I_filt_intpl, window_size)

    # Trim the results to the same length
    min_length = min(len(lv_RSA_M_fif_raw), len(lv_RSA_I_fif_raw))
    lv_RSA_M_fif = lv_RSA_M_fif_raw[:min_length]
    lv_RSA_I_fif = lv_RSA_I_fif_raw[:min_length]

    # detrend RSA signals
    lv_RSA_M_fif_detrended = detrend(lv_RSA_M_fif)
    lv_RSA_I_fif_detrended = detrend(lv_RSA_I_fif)

    # calculate full cross-correlation
    ccf_full = np.correlate(lv_RSA_M_fif_detrended, lv_RSA_I_fif_detrended, mode='full')

    # set maximum lag
    maxlag = 1000
    
    # exctact ccf with maximum lag
    num_lags = (len(ccf_full) - 1) // 2
    ccf = ccf_full[num_lags - maxlag : num_lags + maxlag + 1]

    # read zero lag coefficient
    zeroLagCoefficient = ccf[len(ccf) // 2]

    return zeroLagCoefficient, ccf

def rsa_synchrony_adults(adult_0_ibi, adult_1_ibi):
    """
    This function calculates the zero lag coefficient and the full cross-correlation function (ccf)
    between detrended, logarithmic variances of filtered Respiratory Sinus Arrhythmia (RSA) signals 
    from a mother-infant dyad.

    Parameters:
    - mother_ibi (array-like): list of mother's IBIs in ms
    - infant_ibi (array-like): list of infant's IBIs in ms
    
    Returns:
    - zeroLagCoefficient (float): The zero lag coefficient in the cross-correlation function, indicating the degree of synchrony at zero time lag between the RSA signals.
    - ccf (numpy.ndarray): The full cross-correlation function (limited by maxlag) between the detrended RSA signals of the mother and infant.

    Raises:
    - ValueError: If the length of filtered RSA data is insufficient for further processing.
    """

    # load filters
    filt_adult_path = pkg_resources.resource_filename('rsa_drew', 'adult_rsa_5Hz_cLSq.csv')
    filt_adult = pd.read_csv(filt_adult_path).to_numpy().flatten()

    # resample IBI data to 5Hz
    r_0 = resampled_IBI_ts(adult_0_ibi, 5, False)
    r_1 = resampled_IBI_ts(adult_1_ibi, 5, False)

    # get RSA/BPM and filter RSA
    RSA_0, BPM_0 = poly_filter_data_2011(r_0[:, 1], 51, True) 
    RSA_0_filt = convolve(RSA_0, filt_adult, mode='valid')

    RSA_1, BPM_1 = poly_filter_data_2011(r_1[:, 1], 51, True)
    RSA_1_filt = convolve(RSA_1, filt_adult, mode='valid')    

    # interpolate filtered RSA data
    if len(RSA_0_filt) < 2 or len(RSA_1_filt) < 2:
        raise ValueError("! Insufficient length of filtered RSA data")

    f = interp1d(np.arange(len(RSA_0_filt)), RSA_0_filt)
    RSA_0_filt_intpl = f(np.linspace(0, len(RSA_0_filt) - 1, len(r_0)))
    f = interp1d(np.arange(len(RSA_1_filt)), RSA_1_filt)
    RSA_1_filt_intpl = f(np.linspace(0, len(RSA_1_filt) - 1, len(r_0)))

    # calculate log of variance with sliding window
    window_size = 74  # 15 seconds window at 5 Hz sampling rate
    lv_RSA_0_fif_raw = sliding_window_log_var(RSA_0_filt_intpl, window_size)
    lv_RSA_1_fif_raw = sliding_window_log_var(RSA_1_filt_intpl, window_size)

    # Trim the results to the same length
    min_length = min(len(lv_RSA_0_fif_raw), len(lv_RSA_1_fif_raw))
    lv_RSA_0_fif = lv_RSA_0_fif_raw[:min_length]
    lv_RSA_1_fif = lv_RSA_1_fif_raw[:min_length]

    # detrend RSA signals
    lv_RSA_0_fif_detrended = detrend(lv_RSA_0_fif)
    lv_RSA_1_fif_detrended = detrend(lv_RSA_1_fif)

    # calculate full cross-correlation
    ccf_full = np.correlate(lv_RSA_0_fif_detrended, lv_RSA_1_fif_detrended, mode='full')

    # set maximum lag
    maxlag = 1000
    
    # exctact ccf with maximum lag
    num_lags = (len(ccf_full) - 1) // 2
    ccf = ccf_full[num_lags - maxlag : num_lags + maxlag + 1]

    # read zero lag coefficient
    zeroLagCoefficient = ccf[len(ccf) // 2]

    return zeroLagCoefficient, ccf