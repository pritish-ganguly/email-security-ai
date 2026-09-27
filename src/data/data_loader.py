"""
Data loading utilities for the Email Security AI project.
"""

import pandas as pd


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load an email dataset from a CSV file.

    Parameters
    ----------
    file_path : str
        Path to the CSV dataset.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """

    df = pd.read_csv(file_path)

    return df
