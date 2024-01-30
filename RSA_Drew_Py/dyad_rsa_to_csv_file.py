
import pandas as pd
import os

def dyad_rsa_to_csv_file(mother_rsa, infant_rsa, name, dyad_id, output_dir):

    """
    writes dyad rsa data to a csv file.

    params:
    mother_rsa (array-like), infant_rsa (array-like): rsa data for the infant and mother
    name (str): the name of the specific rsa data
    dyad_id (str): dyad identifiers
    output_dir (str): directory where the CSV file will be saved

    Returns:
    str: Path of the saved CSV file, or throws error message if lengths do not match
    """

    if abs(len(mother_rsa) - len(infant_rsa)) > 1:
        print(f"! Lengths of mother and infant RSA do not match. Output cannot be written. [{name}]")
        return

    data_len = min(len(mother_rsa), len(infant_rsa))
    df = pd.DataFrame({'motherRsa': mother_rsa[:data_len], 'infantRsa': infant_rsa[:data_len]})
    file_path = os.path.join(output_dir, f"{dyad_id}_{name}.csv")
    df.to_csv(file_path, index=False, header=True, sep=';')

    return file_path

