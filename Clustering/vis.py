from dbaccess import get_db_entries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ibi_generator import generate_dyad_ibi, plot_dyad_ibi_sequences

# get data points from database
data = get_db_entries()

base_ibi_adult = [ d['dyad_parameters']['base_ibi_adult'] for d in data ]
base_ibi_infant = [ d['dyad_parameters']['base_ibi_infant'] for d in data ]
base_ibi_harmonics = [ d['dyad_parameters']['base_ibi_infant'] / d['dyad_parameters']['base_ibi_adult'] for d in data ]

# data where extreme ibis are excluded
filtered_data = [d for d in data if d['dyad_parameters']['base_ibi_adult'] > 650 and d['dyad_parameters']['base_ibi_adult'] < 848 and d['dyad_parameters']['base_ibi_infant'] < 598 and d['dyad_parameters']['base_ibi_infant'] > 400]
base_ibi_adult_filt = [ d['dyad_parameters']['base_ibi_adult'] for d in filtered_data ]
base_ibi_infant_filt = [ d['dyad_parameters']['base_ibi_infant'] for d in filtered_data ]
base_ibi_harmonics_filt = [ d['dyad_parameters']['base_ibi_infant'] / d['dyad_parameters']['base_ibi_adult'] for d in filtered_data ]

def print_freq_table(data, bins):
    # Creating histograms and getting bin counts
    hist, bins = np.histogram(data, bins=bins)

    histogram_data = sorted(zip(bins[:-1], hist), key=lambda x: x[1], reverse=True)

    # Creating a DataFrame to represent the histogram data
    histogram_df = pd.DataFrame(histogram_data, columns=['Bin edges', 'freq'])

    print(histogram_df)

def print_histograms(adult, infant, harmonics):
    # Creating subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Plotting first histogram
    axs[0].hist(adult, bins=50, color='skyblue', edgecolor='black')  # Adjust the number of bins as needed
    axs[0].set_title('Adult')
    axs[0].set_xlabel('IBI')
    axs[0].set_ylabel('Frequency')
    axs[0].grid(True)

    # Plotting second histogram
    axs[1].hist(infant, bins=50, color='salmon', edgecolor='black')  # Adjust the number of bins as needed
    axs[1].set_title('Infant')
    axs[1].set_xlabel('IBI')
    axs[1].set_ylabel('Frequency')
    axs[1].grid(True)

    # Plotting third histogram
    axs[2].hist(harmonics, bins=25, color='salmon', edgecolor='black')  # Adjust the number of bins as needed
    axs[2].set_title('Infant')
    axs[2].set_xlabel('Harmonic (Infant/Adult)')
    axs[2].set_ylabel('Frequency')
    axs[2].grid(True)

    # Displaying the subplots
    plt.tight_layout()
    plt.show()

def ibi_combinations_by_harmonic(min, max):
    combinations = [(d['dyad_parameters']['base_ibi_infant'], d['dyad_parameters']['base_ibi_adult']) for d in data]
    filtered = [(infant, adult) for (infant, adult) in combinations if infant/adult > min and infant/adult < max ]
    return filtered


dyad = filtered_data[8]['dyad_parameters']
adult = {}
infant = {}
for key, value in dyad.items():
    if 'adult' in key:
        if 'base_ibi' in key:
            adult['base_ibi'] = value
        elif 'freq' in key:
            adult.setdefault('frequencies', []).append(value)
        elif 'weight' in key:
            adult.setdefault('freq_weights', []).append(value)
        elif 'phase' in key:
            adult.setdefault('phase_shifts', []).append(value)
    elif 'infant' in key:
        if 'base_ibi' in key:
            infant['base_ibi'] = value
        elif 'freq' in key:
            infant.setdefault('frequencies', []).append(value)
        elif 'weight' in key:
            infant.setdefault('freq_weights', []).append(value)
        elif 'phase' in key:
            infant.setdefault('phase_shifts', []).append(value)

ibi_adult, ibi_infant = generate_dyad_ibi(500, adult, infant)
plot_dyad_ibi_sequences(ibi_adult, ibi_infant)