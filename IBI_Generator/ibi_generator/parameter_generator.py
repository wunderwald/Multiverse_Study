import numpy as np
import random

def generate_ibi_params():
    """
    Generates randomized physiologically meaningful frequency bands and weights for the ibi_generator based on HRV.

    Returns:
    - frequencies (numpy.ndarray): An array of 16 frequency bands distributed across VLF, LF, and HF ranges.
    - freq_weights (numpy.ndarray): An array of 16 weights corresponding to the frequency bands, indicating their relative importance in HRV.
    - phase_shifts (numpy.ndarray): An array of 16 phase shifts, one for each frequency band.
    - noise_percentage (float): A percentage indicating the noisyness of the data, i.e. how many samples are to be seen as unusable.

    The frequency bands are divided and randomized as follows:
    - Very Low Frequency (VLF): 4 bands between 0.01 Hz and 0.04 Hz
    - Low Frequency (LF): 6 bands between 0.04 Hz and 0.15 Hz
    - High Frequency (HF): 6 bands between 0.15 Hz and 0.4 Hz

    The weights are assigned randomly within specified ranges for each band.
    """

    # Set number of bands per frequency division
    NUM_VLF_BANDS = 4
    NUM_LF_BANDS = 6
    NUM_HF_BANDS = 6
    NUM_TOTAL_BANDS = NUM_VLF_BANDS + NUM_LF_BANDS + NUM_HF_BANDS

    # Randomize frequency bands within each range
    vlf_bands = np.random.uniform(0.01, 0.04, NUM_VLF_BANDS)
    lf_bands = np.random.uniform(0.04, 0.15, NUM_LF_BANDS)
    hf_bands = np.random.uniform(0.15, 0.4, NUM_HF_BANDS)

    # Combine all frequency bands
    frequencies = np.concatenate((vlf_bands, lf_bands, hf_bands))

    # Assign weights to each band
    vlf_weights = np.random.uniform(0.015, 1.0, NUM_VLF_BANDS)
    lf_weights = np.random.uniform(0.004, 0.4, NUM_LF_BANDS)
    hf_weights = np.random.uniform(0.002, 0.2, NUM_HF_BANDS)

    # Combine all weights
    freq_weights = np.concatenate((vlf_weights, lf_weights, hf_weights))

    # Generate random phase shifts for each band
    phase_shifts = np.random.uniform(0, 2 * np.pi, NUM_TOTAL_BANDS)

    # Determine noise level
    signal_is_noisy = random.random() < .5
    noise_percentage = random.uniform(.01, .2) if signal_is_noisy else .0

    return frequencies, freq_weights, phase_shifts, noise_percentage

def generate_dyad_ibi_params():
    '''
    Generates parameter sets for generating IBI sequences for a dyad.

    Returns:
    - tuple: A tuple containing two dictionaries, one with parameters for the adult and one for the infant.
    '''

    # Define base inter-beat intervals for adult and infant in milliseconds
    BASE_IBI_ADULT = 700  
    BASE_IBI_INFANT = 500 

    # Generate frequency bands, weights, and phase shifts for the adult
    frequencies_adult, freq_weights_adult, phase_shifts_adult, noise_percentage_adult = generate_ibi_params()

    # Generate frequency bands, weights, and phase shifts for the infant
    frequencies_infant, freq_weights_infant, phase_shifts_infant, noise_percentage_infant = generate_ibi_params()

    # Create a parameter dictionary for the adult
    adult_params = {
        'base_ibi': BASE_IBI_ADULT,
        'frequencies': frequencies_adult,
        'freq_weights': freq_weights_adult,
        'phase_shifts': phase_shifts_adult,
        'noise_percentage': noise_percentage_adult
    }

    # Create a parameter dictionary for the infant
    infant_params = {
        'base_ibi': BASE_IBI_INFANT,
        'frequencies': frequencies_infant,
        'freq_weights': freq_weights_infant,
        'phase_shifts': phase_shifts_infant,
        'noise_percentage': noise_percentage_infant
    }

    return adult_params, infant_params