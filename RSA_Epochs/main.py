import os
import pandas as pd
from rsa_epoch import rsa_epoch_synchrony
import matplotlib.pyplot as plt

# i/O dirs
inputDir = "../RSA_Drew/rsa_drew/dyad_ibi_data_mother_infant"

# get input dirs
dyads = [d for d in os.listdir(inputDir) if os.path.isdir(os.path.join(inputDir, d))]

# track sync scores
sync_scores = []

# process dyads
for dyad in dyads:
    print(f"## Processing {dyad}")

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
        epoch_length_ms=30000,
        rsa_method='porges_bohrer',
        sync_type='pearson_corr'
    )
    sync_scores.append(sync)

# Create the histogram
plt.hist(sync_scores, bins=50, alpha=0.7, color='blue', edgecolor='black')

# Add titles and labels
plt.title('Distribution of sync scores (30000ms epochs, Porges Bohrer, Pearson correlation)')
plt.xlabel('Synchrony')
plt.ylabel('Frequency')

# Show the plot
plt.show()
    