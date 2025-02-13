"""
This module provides functions for preprocessing and cleaning TIRF SPT data stored in CSV files.
Functions:
    clean_data(df: pd.DataFrame) -> pd.DataFrame:
        Cleans the input DataFrame by performing various operations such as dropping specific rows,
        selecting and renaming columns, converting data types, and dropping rows with NaN values in 'TRACK_ID'.
    load_csv_files(csv_root: str) -> pd.DataFrame:
        Loads and combines multiple CSV files from a specified directory into a single DataFrame.
"""

from pathlib import Path
import pandas as pd

def clean_data(df)->pd.DataFrame:
    """
    Cleans the input DataFrame by performing the following operations:
    1. Drops the first three rows (assumed to be unit information).
    2. Selects specific columns: 'TRACK_ID', 'POSITION_X', 'POSITION_Y', 'POSITION_T', 'FRAME'.
    3. Converts data types of the selected columns:
    4. Drops rows where 'TRACK_ID' is NaN.
    
    Args:
        df (pd.DataFrame): The input DataFrame to be cleaned.
    Returns:
        pd.DataFrame: The cleaned DataFrame with the specified transformations applied.
    """
    
    # Drop rows 0, 1, and 2 they are unit information
    df = df.drop([0, 1, 2])
    df = df.loc[:, ['TRACK_ID', 'POSITION_X', 'POSITION_Y', 'POSITION_T', 'FRAME']]
    df = df.astype({'TRACK_ID': 'int32', 'POSITION_X': 'float32', 'POSITION_Y': 'float32',
                    'POSITION_T': 'float32', 'FRAME': 'int32'})
    df = df.dropna(subset=['TRACK_ID'])
    df = df.rename(columns={'TRACK_ID': 'TrackID', 'POSITION_X': 'X',
                            'POSITION_Y': 'Y', 'POSITION_T': 'T', 'FRAME': 'Frame'})
    return df

def sort_by_frame(df: pd.DataFrame)->pd.DataFrame:
    """
    Sorts the input DataFrame by 'Frame' in ascending order.
    Args:
        df (pd.DataFrame): The input DataFrame to be sorted.
    Returns:
        pd.DataFrame: The sorted DataFrame.
    """
    df = df.groupby('TrackID', observed=True).apply(lambda x: x.sort_values('Frame')).reset_index(drop=True)
    return df

def format_track_id(x:int)->str:
    """
    Formats an integer as a string with leading zeros to ensure a 
    consistent length of three characters.
    """
    return f'{x:04}'


# Define the path to the directory containing the CSV files
def load_csv_files(csv_root)->pd.DataFrame:
    """
    Load and combine multiple CSV files from a specified directory into a single DataFrame.
    Args:
        csv_root (str): The root directory containing the CSV files.
    Returns:
        pd.DataFrame: A DataFrame containing the combined data from all CSV files in the directory.
    """

    path = Path(csv_root)
    csv_files = path.glob('*.csv')

    dfs = []

    for file in csv_files:
        df = pd.read_csv(file, engine='pyarrow')
        df = clean_data(df)
        df = df.sort_values(by='TrackID')
        df = sort_by_frame(df)
        df['FileID'] = file.stem
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df['TrackID'] = df['TrackID'].apply(format_track_id)
    df['UID'] = df['FileID'] + '-' + df['TrackID']

    return df 
