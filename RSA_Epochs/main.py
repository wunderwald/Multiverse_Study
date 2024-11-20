import os
import pandas as pd
from rsa_epoch import rsa_epoch_synchrony
import matplotlib.pyplot as plt

# i/O dirs
inputDir = "../RSA_Drew/rsa_drew/dyad_ibi_data_mother_infant"

# get input dirs
dyads = [d for d in os.listdir(inputDir) if os.path.isdir(os.path.join(inputDir, d))]

# hyperparameter options
epoch_lengths_ms = [30000, 60000]
rsa_methods = ['drew', 'porges_bohrer', 'hf_hrv']
sync_types = ['pearson_corr', 'spearman_corr', 'kendall_corr']

# calculate synchrony distribution for all hyperparameter settings
for epoch_length_ms in epoch_lengths_ms:
    for rsa_method in rsa_methods:
        for sync_type in sync_types:
            # collect sync scores
            sync_scores = []

            # process dyads
            for dyad in dyads:
                print(f"## Processing {dyad} [{epoch_length_ms}ms epochs, {rsa_method} RSA, {sync_type} synchrony]")

                # paths of IBI files
                ibi_a_path = os.path.join(inputDir, dyad, "ECG1", "ibi_ms.csv")
                ibi_b_path = os.path.join(inputDir, dyad, "ECG2", "ibi_ms.csv")

                # load IBI data
                ibi_a = pd.read_csv(ibi_a_path)['ms'].to_numpy().flatten()
                ibi_b = pd.read_csv(ibi_b_path)['ms'].to_numpy().flatten()

                # calculate synchrony
                sync = rsa_epoch_synchrony(
                    ibi_a_ms=ibi_a, 
                    ibi_b_ms=ibi_b, 
                    epoch_length_ms=epoch_length_ms,
                    rsa_method=rsa_method,
                    sync_type=sync_type,
                    dyad_type='adult_infant'
                )
                sync_scores.append(sync)

            # Create the histogram
            plt.hist(sync_scores, bins=50, alpha=0.7, color='blue', edgecolor='black')

            # Add titles and labels
            plt.title(f"Distribution of sync scores [{epoch_length_ms}ms epochs, {rsa_method} RSA, {sync_type}]")
            plt.xlabel('Synchrony score')
            plt.ylabel('Frequency')

            # save the plot
            filename_plot = f"sync-distribution_{epoch_length_ms}ms_{rsa_method}_{sync_type}.png"
            path_plot = f"./plots/{filename_plot}"
            plt.tight_layout() 
            plt.savefig(path_plot)
            plt.clf()
    