from scipy.interpolate import CubicSpline
import numpy as np

def interpolate_ibi_single_range(ibi_sequence, interpolation_range):
    '''
    Applies cubic spline interpolation to replace samples in a specified index range with interpolated values.

    Parameters:
    - ibi_sequence [numpy.ndarray]: a 1-dimensional list of ibi samples
    - interpolation_range [array-like]: a contiguous list of indices to be replaced by interpolated values

    Returns: 
    numpy.ndarray: the ibi sequence with interpolated samples in the specified range
    '''

    # transform ibi sequence to time series
    t = [0]
    ibi_sum = 0
    for i in range(ibi_sequence.shape[0] - 1):
        ibi_sum = ibi_sum + ibi_sequence[i]
        t.append(ibi_sum)
    ibi_t = np.array(t)

    # Compute interpolated values using CubicSpline
    cs = CubicSpline(ibi_t[interpolation_range], ibi_sequence[interpolation_range])
    interpolation_indices = np.linspace(ibi_t[interpolation_range[0]], ibi_t[interpolation_range[-1]], len(interpolation_range))
    interpolated_samples = cs(interpolation_indices)

    # scale interpolated values so that their sum is the same as the sum of the original values while pr
    sum_interpolated_samples = np.sum(interpolated_samples)
    sum_original_samples = np.sum(ibi_sequence[interpolation_range])
    scl = sum_original_samples / sum_interpolated_samples
    interpolated_samples_scaled = interpolated_samples * scl

    # Replace the faulty data range with the interpolated values in time series
    ibi_sequence[interpolation_range] = interpolated_samples_scaled
    
    return ibi_sequence

def interpolate_ibi(ibi_sequence, interpolation_ranges):
    '''
    Applies cubic spline interpolation to replace samples in a specified index ranges with interpolated values.

    Parameters:
    - ibi_sequence [numpy.ndarray]: a 1-dimensional list of ibi samples
    - interpolation_range [array-like]: a list of contiguous lists of indices to be replaced by interpolated values

    Returns: 
    numpy.ndarray: the ibi sequence with interpolated samples in the specified ranges
    '''
    ibi_sequence_intpl = np.copy(ibi_sequence)
    for interpolation_range in interpolation_ranges:
        ibi_sequence_intpl = interpolate_ibi_single_range(ibi_sequence_intpl, interpolation_range)
    return ibi_sequence_intpl