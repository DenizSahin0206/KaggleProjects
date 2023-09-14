"""
Data Loader script.
Load the data and preprocess it such that it is applicable for the machine learning models
"""

import pandas as pd
import numpy as np

def gather_data(path, file_type='csv'):
    if file_type == 'csv':
        df = pd.read_csv(path)
        return df
    elif file_type == 'excel':
        df = pd.read_excel(path)
        return df
    else:
        print(f"FUNCTION 'gather_data' error:\n"
              f"Please enter a file type this function is build for.\n"
              f"Your options are:\n"
              f"'csv' and 'excel'")

df = gather_data()