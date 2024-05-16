import os
import pandas as pd
from ibi_generator import generate_dyad_ibi
from parameter_generator import generate_dyad_ibi_params
from ibi_plotter import plot_dyad_ibi_sequences
from pymongo import MongoClient

DO_PLOT = False
DO_EXPORT = True
ITERATIONS = 1
RECORDING_LENGTH_S = 300  # 5 minutes
#OUTPUT_DIR = './dyad_ibi_data'
OUTPUT_DIR = './adult_dyad_ibi_data'

# db keys
DB_NAME = 'dyads_final'
# DB_COLLECTION = 'brute_force_physiological_ibi'
DB_COLLECTION = 'brute_force_adult_dyad_physiological_ibi'

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

def ibi_from_generated_dyads():
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

def parse_dyad(dyad):
    adult = {}
    infant = {}

    # Structure adult data
    adult['base_ibi'] = dyad['base_ibi_adult']
    adult['frequencies'] = [dyad[key] for key in dyad if 'adult' in key and 'freq' in key]
    adult['freq_weights'] = [dyad[key] for key in dyad if 'adult' in key and 'weight' in key]
    adult['phase_shifts'] = [dyad[key] for key in dyad if 'adult' in key and 'phase' in key]

    # Structure infant data
    infant['base_ibi'] = dyad['base_ibi_infant']
    infant['frequencies'] = [dyad[key] for key in dyad if 'infant' in key and 'freq' in key]
    infant['freq_weights'] = [dyad[key] for key in dyad if 'infant' in key and 'weight' in key]
    infant['phase_shifts'] = [dyad[key] for key in dyad if 'infant' in key and 'phase' in key]

    return adult, infant


def ibi_from_DB():
    # Connect to the local MongoDB 
    client = MongoClient('mongodb://localhost:27017/')
    db = client[DB_NAME]
    collection = db[DB_COLLECTION]

    # get dyads
    dyads = collection.find()[0]['dyads']

    for index, dyad in enumerate(dyads):
        # generate ibi seqs from params
        adult_params, infant_params = parse_dyad(dyad)
        adult_ibi, infant_ibi = generate_dyad_ibi(RECORDING_LENGTH_S, adult_params, infant_params)
        
        if DO_EXPORT:
            # Directory paths
            dyad_dir = os.path.join(OUTPUT_DIR, f'dyad_{index}')
            ecg1_dir = os.path.join(dyad_dir, 'ECG1')
            ecg2_dir = os.path.join(dyad_dir, 'ECG2')

            # Save the IBI sequences to CSV files
            save_ibi_to_csv(adult_ibi, ecg1_dir, 'ibi_ms.csv')
            save_ibi_to_csv(infant_ibi, ecg2_dir, 'ibi_ms.csv')

ibi_from_DB()