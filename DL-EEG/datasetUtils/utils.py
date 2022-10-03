# Author: Konstantinos Patlatzoglou <konspatl@gmail.com>

import os, shutil
from pathlib import Path
from natsort import natsort_key


# ------------------------------------------------GENERAL UTIL METHODS--------------------------------------------------

def create_directory(path):
    if (os.path.exists(path)):
        shutil.rmtree(path)
    os.makedirs(path)

def path_natural_sort(l):
    return sorted(l, key=lambda x: tuple(natsort_key(k) for k in Path(x).parts))


# ---------------------------------------------DATASET PROCESSING METHODS-----------------------------------------------

def check_dataset_consistency(dataset_info):
    """
    Returns True, if dataset_info is consistent with regards to number of subjects, states and export values

    DATASET_INFO (dict):
      'Subjects' (list): list of Subject names
      'States' (list): list of States per subject
      'EEG Files' (list): list of EEG files per subject
      'Export' (Dict): lists of optional exports - (Optional)
      'Montage File' (str): path to EEG montage file - (Optional)

    Args:
        dataset_info (Dict): Dataset Infomration
    """
    no_of_subjects = len(dataset_info['Subjects'])
    no_of_states = len(dataset_info['States'][0])

    # Check No. of Subjects Consistency
    if not all(len(dataset_info[key]) == no_of_subjects for key in dataset_info if key in ['States', 'EEG Files']):
        return False

    # Check No. of States Consistency
    for key in ['States', 'EEG Files']:
        if not all(len(dataset_info[key][i]) == no_of_states for i in range(no_of_subjects)):
            return False

    # If Export Values, Check No. of Values per Export Key
    if 'Export' in dataset_info:
        for key in dataset_info['Export']:
            if not all(len(dataset_info['Export'][key][i]) == no_of_states for i in range(no_of_subjects)):
                return False

    return True
