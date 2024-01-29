import pandas as pd
import os

def number_to_csv(num, name, dyad_id, output_dir):
    """
    stores a single number in a csv file

    Parameters:
    arr (array-like): array to be converted to CSV
    name (str): data descriptor
    dyad_id (str): dyad identifer
    output_dir (str): directory where the CSV file will be saved

    Returns:
    str: The path to the saved CSV file.
    """
    
    df = pd.DataFrame([num], columns=[name])

    file_path = os.path.join(output_dir, f"{dyad_id}_{name}.csv")

    df.to_csv(file_path, index=False)

    return file_path
