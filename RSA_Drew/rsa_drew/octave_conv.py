# convolution with octave, replicating 'valid' mode
def octave_convolution_valid(a, b, octave_instance):
    '''
    Performs convolution using a oct2py octave instance. Trims the edges for 'valid' convolution.

    Parameters:
    - a (array-like): first array
    - b (array-like): second array
    - octave_instance (Oct2Py): instance of Oct2Py

    Returns
    array: convolution result
    '''
    res = octave_instance.conv(a, b)[0]

    start_index = min(len(a), len(b)) - 1
    end_index = -min(len(a), len(b)) + 1

    if end_index == 0:
        return res[start_index:]
    return res[start_index:end_index]