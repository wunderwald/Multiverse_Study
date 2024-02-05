import pkg_resources
import numpy as np
import pandas as pd
import os
import time
from scipy.signal import convolve, detrend
from oct2py import Oct2Py
from scipy.interpolate import interp1d
from .resampled_ibi_ts import resampled_IBI_ts
from .poly_filter_data_2011 import poly_filter_data_2011
from .sliding_window import sliding_window_log_var
from .number_to_csv import number_to_csv
from .arr_to_csv import arr_to_csv

def rsa_synchrony(mother_ibi, infant_ibi, export_steps=False, use_octave=False, octave_instance=None):
    """
    This function calculates the zero lag coefficient and the full cross-correlation function (ccf)
    between detrended, logarithmic variances of filtered Respiratory Sinus Arrhythmia (RSA) signals 
    from a mother-infant dyad.

    Parameters:
    - mother_ibi (array-like): list of mother's IBIs in ms
    - infant_ibi (array-like): list of infant's IBIs in ms
    - export_steps (bool): export data at each processing step
    - use_octave (bool): use octave instance for convolution
    - octave_instance (Oct2Py): instance of octave created by Oct2Py()
    
    Returns:
    - zeroLagCoefficient (float): The zero lag coefficient in the cross-correlation function, indicating the degree of synchrony at zero time lag between the RSA signals.
    - ccf (numpy.ndarray): The full cross-correlation function (limited by maxlag) between the detrended RSA signals of the mother and infant.

    Raises:
    - ValueError: If the length of filtered RSA data is insufficient for further processing.
    """
    # optionally create octave instance
    if use_octave and not octave_instance:
        octave_instance = Oct2Py()

    # export directories for export_steps
    export_dir = './log'
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
    export_ts = f"{time.time()}".replace(".", "_")
    export_subdir = os.path.join(export_dir, export_ts)
    os.makedirs(export_subdir)

    # load filters
    filt_M_path = pkg_resources.resource_filename('rsa_drew', 'adult_rsa_5Hz_cLSq.csv')
    filt_I_path = pkg_resources.resource_filename('rsa_drew', 'child_RSA.csv')
    filt_M = pd.read_csv(filt_M_path).to_numpy().flatten()
    filt_I = pd.read_csv(filt_I_path).to_numpy().flatten()

    # resample IBI data to 5Hz
    r_M = resampled_IBI_ts(mother_ibi, 5, False)
    r_I = resampled_IBI_ts(infant_ibi, 5, False)

    # optionally export data
    if export_steps:
        arr_to_csv(r_M[:, 1], 'r_M', export_ts, export_subdir)
        arr_to_csv(r_I[:, 1], 'r_I', export_ts, export_subdir)

    # get RSA/BPM and filter RSA
    RSA_M, BPM_M = poly_filter_data_2011(r_M[:, 1], 51, True, True, octave_instance) 
    RSA_M_filt = convolve(RSA_M, filt_M, mode='valid')

    RSA_I, BPM_I = poly_filter_data_2011(r_I[:, 1], 51, True, True, octave_instance)
    RSA_I_filt = convolve(RSA_I, filt_I, mode='valid')

    # optionally export data
    if export_steps:
        arr_to_csv(RSA_M, 'RSA_M', export_ts, export_subdir)
        arr_to_csv(RSA_I, 'RSA_I', export_ts, export_subdir)
        arr_to_csv(RSA_M_filt, 'RSA_M_filt', export_ts, export_subdir)
        arr_to_csv(RSA_I_filt, 'RSA_I_filt', export_ts, export_subdir)
        

    # interpolate filtered RSA data
    if len(RSA_M_filt) < 2 or len(RSA_I_filt) < 2:
        raise ValueError("! Insufficient length of filtered RSA data")

    f = interp1d(np.arange(len(RSA_M_filt)), RSA_M_filt)
    RSA_M_filt_intpl = f(np.linspace(0, len(RSA_M_filt) - 1, len(r_M)))
    f = interp1d(np.arange(len(RSA_I_filt)), RSA_I_filt)
    RSA_I_filt_intpl = f(np.linspace(0, len(RSA_I_filt) - 1, len(r_M)))

    # optionally export data
    if export_steps:
        arr_to_csv(RSA_M_filt_intpl, 'RSA_M_filt_intpl', export_ts, export_subdir)
        arr_to_csv(RSA_I_filt_intpl, 'RSA_I_filt_intpl', export_ts, export_subdir)

    # calculate log of variance with sliding window
    window_size = 74  # 15 seconds window at 5 Hz sampling rate
    lv_RSA_M_fif_raw = sliding_window_log_var(RSA_M_filt_intpl, window_size)
    lv_RSA_I_fif_raw = sliding_window_log_var(RSA_I_filt_intpl, window_size)

    # Trim the results to the same length
    min_length = min(len(lv_RSA_M_fif_raw), len(lv_RSA_I_fif_raw))
    lv_RSA_M_fif = lv_RSA_M_fif_raw[:min_length]
    lv_RSA_I_fif = lv_RSA_I_fif_raw[:min_length]

    # optionally export data
    if export_steps:
        arr_to_csv(lv_RSA_M_fif, 'lv_RSA_M_fif', export_ts, export_subdir)
        arr_to_csv(lv_RSA_I_fif, 'lv_RSA_I_fif', export_ts, export_subdir)

    # detrend RSA signals
    lv_RSA_M_fif_detrended = detrend(lv_RSA_M_fif)
    lv_RSA_I_fif_detrended = detrend(lv_RSA_I_fif)

    # optionally export data
    if export_steps:
        arr_to_csv(lv_RSA_M_fif_detrended, 'lv_RSA_M_fif_detrended', export_ts, export_subdir)
        arr_to_csv(lv_RSA_I_fif_detrended, 'lv_RSA_I_fif_detrended', export_ts, export_subdir)

    # calculate full cross-correlation
    ccf_full = np.correlate(lv_RSA_M_fif_detrended, lv_RSA_I_fif_detrended, mode='full')

    # set maximum lag
    maxlag = 1000
    
    # exctact ccf with maximum lag
    num_lags = (len(ccf_full) - 1) // 2
    ccf = ccf_full[num_lags - maxlag : num_lags + maxlag + 1]

    # optionally export data
    if export_steps:
        arr_to_csv(ccf, 'ccf', export_ts, export_subdir)

    # read zero lag coefficient
    zeroLagCoefficient = ccf[len(ccf) // 2]

    # optionally export data
    if export_steps:
        number_to_csv(zeroLagCoefficient, 'zeroLagCoefficient', export_ts, export_subdir)

    return zeroLagCoefficient, ccf