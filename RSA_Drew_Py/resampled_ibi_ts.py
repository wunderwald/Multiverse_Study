import numpy as np

def resampled_IBI_ts(ibi_in, sample_freq, is_sec):

    """
    Resamples the IBI series into an equally spaced time series. The IBI values
    that fall between samples are represented as a weighted representation.

    Parameters:
    ibi_in (array-like): IBI data series
    sample_freq (float): sampling frequency in Hz
    is_sec (bool): indicates if IBI data is in seconds (false if it is in millis)

    Returns:
    array: The resampled IBI time series.
    """

    # Convert the sampling interval to milliseconds
    samp_interval = 1000 / sample_freq

    # Convert IBI data from seconds to milliseconds if needed
    if is_sec:
        ibi_in = ibi_in * 1000

    # Create a cumulative sum of the IBI data
    time_ser = np.cumsum(ibi_in)

    # Calculate the number of points in the resampled series
    num_pts = int(np.ceil(time_ser[-1] / samp_interval))

    # Initialize the output time series
    time_series_out = np.zeros((num_pts - 1, 2))

    marker = 0
    prev_beat = time_ser[0]
    next_beat = time_ser[0]

    # Resample the IBI series
    for i in range(num_pts - 1):
        time_series_out[i, 0] = samp_interval * (i + 1)
        if time_series_out[i, 0] < next_beat:
            time_series_out[i, 1] = ibi_in[marker]
        else:
            rt_side = time_series_out[i, 0] - next_beat
            lt_side = samp_interval - rt_side
            time_series_out[i, 1] = (rt_side / samp_interval) * ibi_in[marker + 1] + (lt_side / samp_interval) * ibi_in[marker]
            marker += 1
            prev_beat = time_ser[marker - 1]
            next_beat = time_ser[marker]

    return time_series_out

