from ibi_generator import generate_dyad_ibi
from parameter_generator import generate_dyad_ibi_params
from ibi_plotter import plot_dyad_ibi_sequences

RECORDING_LENGTH_S = 500 # 5 minutes

try:
    adult_params, infant_params = generate_dyad_ibi_params()
    adult_ibi, infant_ibi = generate_dyad_ibi(RECORDING_LENGTH_S, adult_params, infant_params)
    plot_dyad_ibi_sequences(adult_ibi, infant_ibi)
except ValueError as e:
    print(e)

