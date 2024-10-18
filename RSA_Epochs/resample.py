'''
Transform a discrete list ob inter-beat intervals (IBIs) to a continuous function using cubic spline interpolation.
The continuous function is sampled at a fixed sampling interval. 

by Moritz Wunderwald, 2023
'''

import os
import shutil
import glob
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

def invalidIbiIndices(ibi):
    invalidIndices = []
    for i, sample in enumerate(ibi):
        if sample > 1500 or sample < 100:
            invalidIndices.append(i)
    return invalidIndices


def resample_ibi(ibi_ms, to_sampling_rate=100):

    sampling_interval_ms = 1000 / to_sampling_rate
    
    # create time axis
    t_ms = []
    ibi_sum = 0
    for ibi_sample in ibi_ms:
        t_ms.append(ibi_sum)
        ibi_sum += ibi_sample

    # Create a cubic spline interpolation
    cs = CubicSpline(t_ms, ibi_ms)

    # Generate time grid for interpolation
    t_ms_interpl = np.arange(min(t_ms), max(t_ms), sampling_interval_ms)

    # Get sample values at new grid
    ibi_ms_interpl = cs(t_ms_interpl)

    # optionally scale interpolated ibi
    sum_ibi_original = sum(ibi_ms)
    sum_ibi_interpl = sum(ibi_ms_interpl)
    scl = sum_ibi_original / sum_ibi_interpl
    ibi_ms_interpl_out = ibi_ms_interpl

    return ibi_ms_interpl
