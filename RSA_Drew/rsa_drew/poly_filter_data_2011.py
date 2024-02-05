from scipy.signal import convolve
from oct2py import Oct2Py
from .poly_v import PolyV, PRE_CALCULATED_DEG_3_SIZE_51
from .octave_conv import octave_convolution_valid


def poly_filter_data_2011(data, poly_size=51, pre_calc_filter=True, use_octave=False, octave_instance=None):
    """
    Applies a polynomial filter to the given data

    Parameters:
    data (array-like): The input data to be filtered
    poly_size (int): The size of the polynomial filter
    use_octave (bool): use GNU octave for convolution to get numerical output that is closer to MATLAB
    pre_calc_filter (bool): use filter coefficients calculated by the original MATLAB script (poly_size=51)
    octave_instance (Oct2Py): instance of octave created by Oct2Py()

    Returns:
    tuple: filtered data and the trend component
    """

    if use_octave and not octave_instance:
        octave_instance = Oct2Py()

    polynomial = PRE_CALCULATED_DEG_3_SIZE_51 if pre_calc_filter else PolyV(3, poly_size) 
    trend = convolve(data, polynomial, mode='valid') if not use_octave else octave_convolution_valid(data, polynomial, octave_instance)

    filtered_data = data[len(polynomial) // 2: len(data) - len(polynomial) // 2] - trend

    return filtered_data, trend