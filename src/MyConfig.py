"""
This file contains several accessory functions, used throughout the project, e.g.:
- error messages.
- checking if an input/output file exists.
"""
import os
import numpy as np
import pandas as pd

def check_if_input_file_exists (relative_path_to_input_file):
    """
    Check whether an input file, given by its relative path, exists.
    If the file doesn't exist - exit with a proper error msg.
    """
    if not (os.path.isfile (relative_path_to_input_file)):
        error (f'the input file {relative_path_to_input_file} does not exist')

def error (str):
    print ('error: {}' .format (str))
    exit ()


