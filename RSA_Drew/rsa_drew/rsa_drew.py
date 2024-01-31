import pkg_resources
import numpy as np
import pandas as pd
from scipy.signal import convolve, detrend
from scipy.interpolate import interp1d
from .resampled_ibi_ts import resampled_IBI_ts
from .poly_filter_data_2011 import poly_filter_data_2011
from .sliding_window import sliding_window_log_var

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

    # load filters
    # filt_M = pd.read_csv('adult_rsa_5Hz_cLSq.csv').to_numpy().flatten()
    # filt_I = pd.read_csv('child_RSA.csv').to_numpy().flatten()
    filt_M_path = pkg_resources.resource_filename('rsa_drew', 'adult_rsa_5Hz_cLSq.csv')
    filt_I_path = pkg_resources.resource_filename('rsa_drew', 'child_RSA.csv')
    filt_M = pd.read_csv(filt_M_path).to_numpy().flatten()
    filt_I = pd.read_csv(filt_I_path).to_numpy().flatten()

    # resamle IBI data to 5Hz
    r_M = resampled_IBI_ts(mother_ibi, 5, False)
    r_I = resampled_IBI_ts(infant_ibi, 5, False)

    # get RSA/BPM and filter RSA
    RSA_M, BPM_M = poly_filter_data_2011(r_M[:, 1], 51) 
    RSA_M_filt = convolve(RSA_M, filt_M, mode='valid')

    RSA_I, BPM_I = poly_filter_data_2011(r_I[:, 1], 51)
    RSA_I_filt = convolve(RSA_I, filt_I, mode='valid')

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