from ibi_generator import generate_ibi_sequence
from parameter_generator import generate_ibi_params
from ibi_plotter import plot_ibi_sequence

# base inter beat intervals in ms
BASE_IBI_ADULT = 700
BASE_IBI_INFANT = 500

# number of samples to be generated
NUM_SAMPLES = 100

# generate ibi seqence
try:
    freq_bands, freq_weights, phase_shifts = generate_ibi_params()
    ibi_sequence = generate_ibi_sequence(NUM_SAMPLES, BASE_IBI_ADULT, freq_bands, freq_weights, phase_shifts)
    plot_ibi_sequence(ibi_sequence)
except ValueError as e:
    print(e)

# TODO dyad_ibi_generator
# return base_ibi_infant, base_ibi_adult, freq_weights_adult....
# generate sequences for infant and adult shorten the longer sequence so that the sum difference is minimal [ or so]