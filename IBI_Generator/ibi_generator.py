import numpy as np

def generate_ibi_sequence(num_samples, base_ibi, frequencies, freq_weights, phase_shifts):
    """
    Generates an IBI (Inter-Beat-Interval) sequence.

    Parameters:
    - num_samples (int): Number of samples in the output sequence.
    - base_ibi (float): Base inter-beat interval in milliseconds. This is used to initialize the output sequence.
    - frequencies (array-like): An array of 16 frequencies in Hz that are used to manipulate the base_ibi sequence.
    - freq_weights (array-like): Weights for each frequency band.
    - phase_shifts (array-like): An array of 16 phase shifts, one for each frequency band.

    Returns:
    - numpy.ndarray: An array representing the IBI sequence.

    Raises:
    - ValueError: If the length of frequencies or freq_weights is not 16, or if any weight is <= 0.
    """

    # Parameter checks
    if len(frequencies) != 16 or len(freq_weights) != 16 or len(phase_shifts) != 16:
        raise ValueError("frequencies, freq_weights and phase_shifts must all have exactly 16 elements.")
    
    if any(weight <= 0 for weight in freq_weights):
        raise ValueError("All freq_weights must be greater than 0.")

    # Time for each sample (assuming constant IBI initially)
    times = np.arange(num_samples) * base_ibi

    # Initialize the IBI sequence
    ibi_sequence = np.full(num_samples, base_ibi, dtype=float)

    # Iterate over each frequency band
    for freq, weight, phase_shift in zip(frequencies, freq_weights, phase_shifts):
        # Calculate the sine wave for this frequency
        sine_wave = (np.sin(2 * np.pi * freq * times / 1000 + phase_shift)) * 1/64

        # Scale the sine wave by its weight
        scaled_sine_wave = sine_wave * weight

        # Multiply the IBI sequence by the scaled sine wave (1 + value) to adjust the base IBI
        ibi_sequence *= (1 + scaled_sine_wave)

    return ibi_sequence

def generate_dyad_ibi(recording_time_s, adult_params, infant_params):
    '''
    Generates IBI sequences for a dyad (an adult and an infant) based on given parameters.

    Parameters:
    - recording_time_s (float): Total recording time in seconds.
    - adult_params (dict): Dictionary of parameters for the adult's IBI sequence. 
                           Must include 'base_ibi', 'frequencies', 'freq_weights', and 'phase_shifts'.
    - infant_params (dict): Dictionary of parameters for the infant's IBI sequence.
                            Must include 'base_ibi', 'frequencies', 'freq_weights', and 'phase_shifts'.

    Returns:
    - tuple: A tuple containing two numpy arrays, one for the adult's IBI sequence and one for the infant's IBI sequence.

    Raises:
    - ValueError: If there's an issue with the parameters or with generating the IBI sequence.
    '''

    required_keys = ['base_ibi', 'frequencies', 'freq_weights', 'phase_shifts']

    # Check for required keys in parameters
    for key in required_keys:
        if key not in adult_params or key not in infant_params:
            raise ValueError(f'Missing required parameter: {key}')

    # Convert recording time to milliseconds
    recording_time_ms = recording_time_s * 1000

    # Determine num_samples so that recording time is definitely exceeded
    min_base_ibi = min(adult_params['base_ibi'], infant_params['base_ibi'])
    num_samples = recording_time_ms // min_base_ibi * 2

    try:
        # Create IBI sequences for the dyad
        adult_ibi_full = generate_ibi_sequence(
            num_samples, 
            adult_params['base_ibi'], 
            adult_params['frequencies'],
            adult_params['freq_weights'],
            adult_params['phase_shifts']
        )
        infant_ibi_full = generate_ibi_sequence(
            num_samples, 
            infant_params['base_ibi'], 
            infant_params['frequencies'],
            infant_params['freq_weights'],
            infant_params['phase_shifts']
        )
    except ValueError as e:
        raise ValueError(f"Error generating IBI sequence: {e}")

    # Crop adult IBI sequence to fit the recording length
    adult_ibi = np.array([])
    ibi_sum_adult = 0
    for ibi_sample in adult_ibi_full:
        ibi_sum_adult += ibi_sample
        if ibi_sum_adult > recording_time_ms:
            break
        adult_ibi = np.append(adult_ibi, ibi_sample)

    # Crop infant IBI sequence to fit the recording length
    infant_ibi = np.array([])
    ibi_sum_infant = 0
    for ibi_sample in infant_ibi_full:
        ibi_sum_infant += ibi_sample
        if ibi_sum_infant > recording_time_ms:
            break
        infant_ibi = np.append(infant_ibi, ibi_sample)

    return adult_ibi, infant_ibi