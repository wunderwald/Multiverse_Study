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