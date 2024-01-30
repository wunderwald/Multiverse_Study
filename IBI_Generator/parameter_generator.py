import numpy as np

def generate_ibi_generator_params():
    """
    Generates randomized physiologically meaningful frequency bands and weights for the ibi_generator based on HRV.

    Returns:
    - freq_bands (numpy.ndarray): An array of 16 frequency bands distributed across VLF, LF, and HF ranges.
    - freq_weights (numpy.ndarray): An array of 16 weights corresponding to the frequency bands, indicating their relative importance in HRV.

    The frequency bands are divided and randomized as follows:
    - Very Low Frequency (VLF): 4 bands between 0.01 Hz and 0.04 Hz
    - Low Frequency (LF): 6 bands between 0.04 Hz and 0.15 Hz
    - High Frequency (HF): 6 bands between 0.15 Hz and 0.4 Hz

    The weights are assigned randomly within specified ranges for each band:
    - VLF weights: 0.1 to 0.5
    - LF weights: 0.5 to 1.5
    - HF weights: 0.5 to 1.5
    """

    # Set number of bands per frequency division
    NUM_VLF_BANDS = 4
    NUM_LF_BANDS = 6
    NUM_HF_BANDS = 6

    # Randomize frequency bands within each range
    vlf_bands = np.random.uniform(0.01, 0.04, NUM_VLF_BANDS)
    lf_bands = np.random.uniform(0.04, 0.15, NUM_LF_BANDS)
    hf_bands = np.random.uniform(0.15, 0.4, NUM_HF_BANDS)

    # Combine all frequency bands
    freq_bands = np.concatenate((vlf_bands, lf_bands, hf_bands))

    # Assign weights to each band
    vlf_weights = np.random.uniform(0.1, 0.5, NUM_VLF_BANDS)
    lf_weights = np.random.uniform(0.5, 1.5, NUM_LF_BANDS)
    hf_weights = np.random.uniform(0.5, 1.5, NUM_HF_BANDS)

    # Combine all weights
    freq_weights = np.concatenate((vlf_weights, lf_weights, hf_weights))

    return freq_bands, freq_weights
