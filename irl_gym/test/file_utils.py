"""
Functions for importing and exporting files.
"""

__license__ = "BSD-3"
__docformat__ = 'reStructuredText'
__author__ = "Jared Beard"

import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import pandas as pd

__all__ = ["import_file", "export_file", "create_file", "merge_files"]

def import_file(filepath : str):
    """
    Import a file as a Pandas dataframe based on its file extension

    :param filepath: (str) file path
    :return: (pandas.DataFrame) the imported dataframe
    """
    extension = filepath.split(".")[-1]
    if not os.path.isfile(filepath):
        return pd.DataFrame()
    if extension == "csv":
        return pd.read_csv(filepath)
    elif extension == "xlsx":
        return pd.read_excel(filepath)
    elif extension == "json":
        return pd.read_json(filepath)
    else:
        raise ValueError(f"Unsupported file type: {extension}")

def export_file(df : pd.DataFrame, filepath : str):
    """
    Write a Pandas dataframe to a file based on its file extension

    :param df: (pandas.DataFrame) the dataframe to be written
    :param filepath: (str) file path
    :return: None
    """
    extension = filepath.split(".")[-1]
    if extension == "csv":
        df.to_csv(filepath, index=False)
    elif extension == "xlsx":
        df.to_excel(filepath, index=False)
    elif extension == "json":
        df.to_json(filepath, index=False)
    else:
        raise ValueError(f"Unsupported file type: {extension}")

def create_file(file_path : str):
    """
    Checks for the existence of file and creates an empty frame if not

    :param file_path: (str) Desired file
    :return: (pd.DataFrame) Data
    """
    file_extension = os.path.splitext(file_path)[1]
    if not os.path.exists(file_path):
        empty_df = pd.DataFrame()
        if file_extension == '.csv':
            empty_df.to_csv(file_path, index=False)
        elif file_extension == '.json':
            empty_df.to_json(file_path)
        elif file_extension == '.xlsx':
            empty_df.to_excel(file_path, index=False)
        else:
            return "Invalid file type"
        return "Empty file created at: " + file_path
    return "File already exists at: " + file_path

def merge_files(file_list : str):
    """
    Merges the data from a list of files into a single pd.DataFrame and saves them at the last file.
    
    :param file_list: A list of file names to merge.
    """
    data = pd.DataFrame()
    for file in file_list:
        data = pd.concat([data, import_file(file)])
    export_file(data, file_list[-1])