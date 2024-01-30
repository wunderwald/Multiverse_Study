import os

from IBI_Generator.ibi_generator import generate_dyad_ibi
from RSA_Drew.rsa_drew import rsa_drew

# I/O dirs
inputDir = "../RSA_Drew_Py/dyad_ibi_data"

# get input dirs
dyads = [d for d in os.listdir(inputDir) if os.path.isdir(os.path.join(inputDir, d))]

# process dyads
for dyad in dyads:
    print(f"## Processing {dyad}")