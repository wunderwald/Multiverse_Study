import numpy as np

def sliding_window_log_var(data, window_size):
    '''
    calculates the natural log of the variance in a sliding window

    Parameters:
    data (array-like): data that sliding window operation is aplied to
    window_size (int): window size

    Returns:
    np.array: the output array
    '''
    
    half_window = window_size // 2
    log_var = []

    for i in range(half_window, len(data) - half_window):
        window_data = data[i - half_window:i + half_window]
        log_var.append(np.log(np.var(window_data)))

    return np.array(log_var)