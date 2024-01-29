import os
import pandas as pd

def arr_to_csv(arr, name, dyad_id, output_dir):
    """
    Converts an array to a CSV file.

    Parameters:
    arr (array-like): The array to be converted to CSV.
    name (str): Name for the CSV header.
    dyad_id (str): Dyad identifier to be used in the filename.
    output_dir (str): Directory where the CSV file will be saved.

    Returns:
    str: The path to the saved CSV file.
    """

    df = pd.DataFrame(arr, columns=[name])
    file_path = os.path.join(output_dir, f"{dyad_id}_{name}.csv")
    df.to_csv(file_path, index=False)

    return file_path