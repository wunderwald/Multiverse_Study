import os
import shutil
import numpy as np
import pandas as pd
from scipy.signal import convolve, detrend
from scipy.interpolate import interp1d
from resampled_ibi_ts import resampled_IBI_ts
from poly_filter_data_2011 import poly_filter_data_2011
from dyad_rsa_to_csv_file import dyad_rsa_to_csv_file
from number_to_csv import number_to_csv
from arr_to_csv import arr_to_csv
from sliding_window import sliding_window_log_var

# I/O dirs
inputDir = "./dyad_ibi_data"
outputDir = "./dyad_rsa_data"

# create output dir if it doesn't exist yet
if not os.path.exists(outputDir):
    os.makedirs(outputDir)

# clear output dir
for f in os.listdir(outputDir):
    file_path = os.path.join(outputDir, f)
    if os.path.isfile(file_path):
        os.remove(file_path)
    elif os.path.isdir(file_path):
        shutil.rmtree(file_path)

# get input dirs
dyads = [d for d in os.listdir(inputDir) if os.path.isdir(os.path.join(inputDir, d))]

# process dyads
for dyad in dyads:
    print(f"## Processing {dyad}")

    # paths of IBI files
    motherPath = os.path.join(inputDir, dyad, "ECG1", "ibi_ms.csv")
    infantPath = os.path.join(inputDir, dyad, "ECG2", "ibi_ms.csv")

    # load IBI data
    M = pd.read_csv(motherPath)['ms'].to_numpy().flatten()
    I = pd.read_csv(infantPath)['ms'].to_numpy().flatten()

    # load filters
    filt_M = pd.read_csv('adult_rsa_5Hz_cLSq.csv').to_numpy().flatten()
    filt_I = pd.read_csv('child_RSA.csv').to_numpy().flatten()

    # resamle IBI data to 5Hz
    r_M = resampled_IBI_ts(M, 5, False)
    r_I = resampled_IBI_ts(I, 5, False)

    # get RSA/BPM and filter RSA
    RSA_M, BPM_M = poly_filter_data_2011(r_M[:, 1], 51) 
    RSA_M_filt = convolve(RSA_M, filt_M, mode='valid')

    RSA_I, BPM_I = poly_filter_data_2011(r_I[:, 1], 51)
    RSA_I_filt = convolve(RSA_I, filt_I, mode='valid')

    # interpolate filtered RSA data
    if len(RSA_M_filt) < 2 or len(RSA_I_filt) < 2:
        print("! Insufficient length of filtered RSA data")
        continue

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

    # make subdir
    outputSubdir = os.path.join(outputDir, dyad)
    os.makedirs(outputSubdir)

    # export data
    number_to_csv(zeroLagCoefficient, "zeroLagCoefficient", dyad, outputSubdir)
    arr_to_csv(ccf, "ccf", dyad, outputSubdir)
    dyad_rsa_to_csv_file(lv_RSA_M_fif_detrended, lv_RSA_I_fif_detrended, "detrended", dyad, outputSubdir)
    