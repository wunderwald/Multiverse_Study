
import os
import shutil
import pandas as pd
import rsa_drew as rsa
from oct2py import Oct2Py

# make octave instance
octave_instance = Oct2Py()

# I/O dirs
inputDir = '../RSA_Drew/rsa_drew/dyad_ibi_data'
outputDir = "./dyad_rsa_data"

# create output dir if it doesn't exist yet
if not os.path.exists(outputDir):
    os.makedirs(outputDir)

# clear output dir
for f in os.listdir(outputDir):
    file_path = os.path.join(outputDir, f)
    if os.path.isfile(file_path):
        os.remove(file_path)
    elif os.path.isdir(file_path):
        shutil.rmtree(file_path)

# get input dirs
dyads = [d for d in os.listdir(inputDir) if os.path.isdir(os.path.join(inputDir, d))]

# process dyads
for dyad in dyads:
    print(f"## Processing {dyad}")

    # paths of IBI files
    motherPath = os.path.join(inputDir, dyad, "ECG1", "ibi_ms.csv")
    infantPath = os.path.join(inputDir, dyad, "ECG2", "ibi_ms.csv")

    # load IBI data
    M = pd.read_csv(motherPath)['ms'].to_numpy().flatten()
    I = pd.read_csv(infantPath)['ms'].to_numpy().flatten()

    zcl, ccf = rsa.rsa_synchrony(
        mother_ibi=M, 
        infant_ibi=I, 
        export_steps=True,
        use_octave=True,
        octave_instance=octave_instance)

    print(zcl)