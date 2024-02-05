from scipy.signal import convolve
from .poly_v import PolyV, PRE_CALCULATED_DEG_3_SIZE_51


def poly_filter_data_2011(data, poly_size=51, pre_calc_filter=True):
    """
    Applies a polynomial filter to the given data

    Parameters:
    data (array-like): The input data to be filtered
    poly_size (int): The size of the polynomial filter
    pre_calc_filter (bool): use filter coefficients calculated by the original MATLAB script (poly_size=51)

    Returns:
    tuple: filtered data and the trend component
    """

    polynomial = PRE_CALCULATED_DEG_3_SIZE_51 if pre_calc_filter else PolyV(3, poly_size) 
    trend = convolve(data, polynomial, mode='valid')

    filtered_data = data[len(polynomial) // 2: len(data) - len(polynomial) // 2] - trend

    return filtered_data, trend