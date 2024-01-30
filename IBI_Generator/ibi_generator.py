import numpy as np

def generate_ibi_sequence(num_samples, base_ibi, freq_bands, freq_weights, phase_shifts):
    """
    Generates an IBI (Inter-Beat-Interval) sequence.

    Parameters:
    - num_samples (int): Number of samples in the output sequence.
    - base_ibi (float): Base inter-beat interval in milliseconds. This is used to initialize the output sequence.
    - freq_bands (array-like): An array of 16 frequencies in Hz that are used to manipulate the base_ibi sequence.
    - freq_weights (array-like): Weights for each frequency band.
    - phase_shifts (array-like): An array of 16 phase shifts, one for each frequency band.

    Returns:
    - numpy.ndarray: An array representing the IBI sequence.

    Raises:
    - ValueError: If the length of freq_bands or freq_weights is not 16, or if any weight is <= 0.
    """

    # Parameter checks
    if len(freq_bands) != 16 or len(freq_weights) != 16 or len(phase_shifts) != 16:
        raise ValueError("freq_bands, freq_weights and phase_shifts must all have exactly 16 elements.")
    
    if any(weight <= 0 for weight in freq_weights):
        raise ValueError("All freq_weights must be greater than 0.")

    # Time for each sample (assuming constant IBI initially)
    times = np.arange(num_samples) * base_ibi

    # Initialize the IBI sequence
    ibi_sequence = np.full(num_samples, base_ibi, dtype=float)

    # Iterate over each frequency band
    for freq, weight, phase_shift in zip(freq_bands, freq_weights, phase_shifts):
        # Calculate the sine wave for this frequency
        sine_wave = (np.sin(2 * np.pi * freq * times / 1000 + phase_shift)) * 1/64

        # Scale the sine wave by its weight
        scaled_sine_wave = sine_wave * weight

        # Multiply the IBI sequence by the scaled sine wave (1 + value) to adjust the base IBI
        ibi_sequence *= (1 + scaled_sine_wave)

    return ibi_sequence

