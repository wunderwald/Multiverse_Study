import ibi_generator as ibi
import rsa_drew as rsa

RECORDING_TIME = 300

adult_params, infant_params = ibi.generate_dyad_ibi_params()
adult_ibi, infant_ibi = ibi.generate_dyad_ibi(RECORDING_TIME, adult_params, infant_params)

zlc, ccf = rsa.rsa_synchrony(adult_ibi, infant_ibi)

print(zlc)