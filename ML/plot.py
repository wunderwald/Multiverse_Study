import matplotlib.pyplot as plt
import os

def plot_fitness_distribution(fitness_values, generation_number, folder_path):
    """
    Creates and saves a plot of the distribution of fitness values to a specified folder.

    Parameters:
    - fitness_values (list or np.array): An array or list of fitness values for which the distribution is to be plotted.
    - generation_number (int): The generation number, used in naming the plot file.
    - folder_path (str): The path to the folder where the plot will be saved.

    Notes:
    - The function creates a histogram of the fitness values.
    - The plot is saved as a PNG file with a name indicating the generation number.
    - If the specified folder does not exist, it is created.
    """
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Create the plot
    plt.figure()
    plt.hist(fitness_values, bins=20, alpha=0.75)
    plt.title(f'Fitness Distribution - Generation {generation_number}')
    plt.xlabel('Fitness Value')
    plt.ylabel('Frequency')

    # Save the plot
    filename = os.path.join(folder_path, f'fitness_distribution_gen_{generation_number}.png')
    plt.savefig(filename)

    # Close the plot to free up memory
    plt.close()