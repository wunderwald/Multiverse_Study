
import numpy as np
from scipy.signal import convolve
from poly_v import PolyV

def PolyFilterData_2011(data, poly_size):

    """
    Applies a polynomial filter to the given data

    Parameters:
    data (array-like): The input data to be filtered
    poly_size (int): The size of the polynomial filter

    Returns:
    tuple: filtered data and the trend component
    """

    polynomial = PolyV(3, poly_size)
    trend = convolve(data, polynomial, mode='valid')
    filtered_data = data[len(polynomial) // 2: len(data) - len(polynomial) // 2] - trend

    return filtered_data, trend