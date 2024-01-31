import ibi_generator as ibi
import rsa_drew as rsa
import numpy as np

RECORDING_TIME = 300

zlcs = []

for i in range(100):
    adult_params, infant_params = ibi.generate_dyad_ibi_params()
    adult_ibi, infant_ibi = ibi.generate_dyad_ibi(RECORDING_TIME, adult_params, infant_params)

    zlc, ccf = rsa.rsa_synchrony(adult_ibi, infant_ibi)

    zlcs.append(zlc)

print(f"Min: {np.min(zlcs)}")
print(f"Max: {np.max(zlcs)}")
print(f"Mean: {np.mean(zlcs)}")
print(f"Median: {np.median(zlcs)}")