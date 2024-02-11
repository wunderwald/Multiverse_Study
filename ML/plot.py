import matplotlib.pyplot as plt
import os
import shutil

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
    plt.hist(fitness_values, bins=100, alpha=0.75)
    plt.title(f'Fitness Distribution - Generation {generation_number}')
    plt.xlabel('Fitness Value')
    plt.ylabel('Frequency')

    # Save the plot
    filename = os.path.join(folder_path, f'fitness_distribution_gen_{generation_number}.png')
    plt.savefig(filename)

    # Close the plot to free up memory
    plt.close()

def clear_folder(folder_path):
    """
    Clears all files and subdirectories in a specified folder without deleting the folder itself.

    Parameters:
    - folder_path (str): The path to the folder to be cleared.

    Notes:
    - If the folder does not exist, the function does nothing and prints a message indicating this.
    - The function removes all files and subdirectories within the specified folder but keeps the folder itself intact.
    """
    # Check if the folder exists
    if not os.path.exists(folder_path):
        return

    # Iterate over all files and directories in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Check if it's a file or directory
        if os.path.isfile(file_path) or os.path.islink(file_path):
            # If it's a file or a link, delete it
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            # If it's a directory, delete it and all its contents
            shutil.rmtree(file_path)