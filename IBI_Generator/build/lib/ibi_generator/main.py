import os
import pandas as pd
from ibi_generator import generate_dyad_ibi
from parameter_generator import generate_dyad_ibi_params
from ibi_plotter import plot_dyad_ibi_sequences

DO_PLOT = True
DO_EXPORT = False
ITERATIONS = 1

RECORDING_LENGTH_S = 300  # 5 minutes
OUTPUT_DIR = './dyad_ibi_data'

def save_ibi_to_csv(ibi_sequence, directory, filename):
    """
    Saves the IBI sequence to a CSV file.

    Parameters:
    - ibi_sequence (array-like): The IBI sequence to save.
    - directory (str): The directory where the CSV file will be saved.
    - filename (str): The name of the CSV file.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    filepath = os.path.join(directory, filename)
    pd.DataFrame({'ms': ibi_sequence}).to_csv(filepath, index=False)

try:
    for i in range(ITERATIONS):
        adult_params, infant_params = generate_dyad_ibi_params()
        adult_ibi, infant_ibi = generate_dyad_ibi(RECORDING_LENGTH_S, adult_params, infant_params)
        
        if DO_EXPORT:
            # Directory paths
            dyad_dir = os.path.join(OUTPUT_DIR, f'dyad_{i}')
            ecg1_dir = os.path.join(dyad_dir, 'ECG1')
            ecg2_dir = os.path.join(dyad_dir, 'ECG2')

            # Save the IBI sequences to CSV files
            save_ibi_to_csv(adult_ibi, ecg1_dir, 'ibi_ms.csv')
            save_ibi_to_csv(infant_ibi, ecg2_dir, 'ibi_ms.csv')
        
        if DO_PLOT:
            plot_dyad_ibi_sequences(adult_ibi, infant_ibi)


except ValueError as e:
    print(f"Error encountered: {e}")
