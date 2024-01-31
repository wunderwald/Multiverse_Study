import matplotlib.pyplot as plt

def plot_ibi_sequence(ibi_sequence):
    """
    Plots the IBI sequence.

    Parameters:
    - ibi_sequence (array-like): An array representing the IBI sequence to be plotted.
    """

    plt.figure(figsize=(12, 6))
    plt.plot(ibi_sequence, label='IBI Sequence')
    plt.xlabel('Sample Index')
    plt.ylabel('IBI (ms)')
    plt.title('Inter-Beat-Interval (IBI) Sequence')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_dyad_ibi_sequences(adult_ibi, infant_ibi):
    """
    Plots IBI sequences for a dyad.

    Parameters:
    - adult_ibi (array-like): The IBI sequence for the adult.
    - infant_ibi (array-like): The IBI sequence for the infant.
    """

    # Create a figure with two subplots (2 rows, 1 column)
    _, axs = plt.subplots(2, 1, figsize=(12, 6))

    # Plotting the adult's IBI sequence in the first subplot
    axs[0].plot(adult_ibi, label='Adult IBI Sequence')
    axs[0].set_xlabel('Sample Index')
    axs[0].set_ylabel('IBI (ms)')
    axs[0].set_title('Adult IBI Sequence')
    axs[0].legend()
    axs[0].grid(True)

    # Plotting the infant's IBI sequence in the second subplot
    axs[1].plot(infant_ibi, label='Infant IBI Sequence')
    axs[1].set_xlabel('Sample Index')
    axs[1].set_ylabel('IBI (ms)')
    axs[1].set_title('Infant IBI Sequence')
    axs[1].legend()
    axs[1].grid(True)

    # Adjust layout
    plt.tight_layout()
    plt.show()
