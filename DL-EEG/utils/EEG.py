# Author: Konstantinos Patlatzoglou <konspatl@gmail.com>

import numpy as np
import json
from sklearn import preprocessing

# ------------------------------------------------- EEG UTIL -----------------------------------------------------------

EEG_2D_grid_map = {            'Fp1':(0,1),            'Fp2':(0,3),
                   'F7':(1,0), 'F3':(1,1), 'Fz':(1,2), 'F4':(1,3), 'F8':(1,4),
                   'T3':(2,0), 'C3':(2,1), 'Cz':(2,2), 'C4':(2,3), 'T4':(2,4),
                   'T7':(2,0),                                     'T8':(2,4),
                   'T5':(3,0), 'P3':(3,1), 'Pz':(3,2), 'P4':(3,3), 'T6':(3,4),
                   'P7':(3,0),                                     'P8':(3,4),
                               'O1':(4,1), 'Oz':(4,2), 'O2':(4,3)
                  }

EEG_10_20_compatible = ['F7', 'T3', 'T7', 'T5', 'P7', 'Fp1', 'F3', 'C3', 'P3', 'O1', 'Fz', 'Cz', 'Pz', 'Oz', 'Fp2',
                        'F4', 'C4', 'P4', 'O2', 'F8', 'T4', 'T8', 'T6', 'P8']

# -------------------------------------------- EEG DATASET METHODS------------------------------------------------------

def find_dataset_path(datasets_path, EEG_DATASET):
    """
    Returns the directory of the EEG dataset described in EEG_DATASET.
    This method checks for the EEG_DATASET_PARAMETERS json file and returns the directory if the dataset descriptions
    are consistent with respect to studies, subjects, states and targets (exports)

    EEG_DATASET (dict):
        'Study' (str): the name of the study
        'Subjects' (list): list of subject names
        'States' (list): list of state names
        'Targets' (str of list): string or list with Target names
        'Target Values' (str or list): 'Read_from_info_file' or list of Targets

    Args:
        datasets_path (path): the directory of the dataset subdirectories
        EEG_DATASET (dict): the selected EEG dataset description
    Returns:
        dataset_path (path): the directory of the selected dataset (path)
    """
    dataset_path = None

    # List of All Dataset Directories
    datasets_path_dirs = sorted([x for x in datasets_path.iterdir() if x.is_dir()])

    for sub_dir in datasets_path_dirs:  # For each Directory
        try: # Open EEG_DATASET_PARAMETERS file
            with open(str(sub_dir / 'EEG_DATASET_PARAMETERS.json')) as json_file:
                eeg_dataset_parameters = json.load(json_file)
        except FileNotFoundError:
            continue

        # Check Study
        if  eeg_dataset_parameters['EEG_DATASET']['Study'] != EEG_DATASET['Study']:
            continue

        # Check Subjects
        if not all([subject in eeg_dataset_parameters['EEG_DATASET']['Subjects'] for subject in EEG_DATASET['Subjects']]):
            continue

        # Check States
        if not all([state in eeg_dataset_parameters['EEG_DATASET']['States'] for state in EEG_DATASET['States']]):
            continue

        # Check if Targets (list case) exist in file (Export)
        if EEG_DATASET['Target Values'] == 'Read_from_info_file' and isinstance(EEG_DATASET['Targets'], list):
            if not 'Export' in eeg_dataset_parameters['EEG_DATASET']:
                continue
            if not all([target in eeg_dataset_parameters['EEG_DATASET']['Export'] for target in EEG_DATASET['Targets']]):
                continue

        dataset_path = sub_dir
        break

    return dataset_path


def get_dataset(dataset_path):
    """
    Creates a Raw EEG Dataset structure from the given directory

    Args:
        dataset_path (Path): Path to the directory of an Exported Dataset
    Returns:
        raw_eeg_dataset (Dict): Structure including 'EEG Dataset Parameters', 'Subjects', 'EEG Data' and 'Info'
    """
    try:  # Open EEG_DATASET_PARAMETERS file
        with open(str(dataset_path / 'EEG_DATASET_PARAMETERS.json')) as json_file:
            eeg_dataset_parameters = json.load(json_file)
    except FileNotFoundError:
        print('No EEG_DATASET_PARAMETERS.json file Found')
        return None

    subjects = []
    eeg_data_list = []
    subject_info_list = []

    eeg_data_files_list = sorted(list(dataset_path.glob('*eeg_data.npy')))
    for eeg_data_file in eeg_data_files_list: # For all .npy files

        subject = eeg_data_file.name.replace('_eeg_data.npy', '')
        eeg_data = np.load(str(eeg_data_file), allow_pickle=True)
        subject_info = json.load(open(str(dataset_path / (subject + '_info.json'))))

        subjects.append(subject)
        eeg_data_list.append(eeg_data)
        subject_info_list.append(subject_info)

    raw_eeg_dataset = {'EEG Dataset Parameters': eeg_dataset_parameters, 'Subjects': np.array(subjects),
                       'EEG Data': np.array(eeg_data_list), 'Info': subject_info_list}

    return raw_eeg_dataset


def select_data_from_dataset(raw_eeg_dataset, EEG_DATASET, id=0):
    """
    Creates an EEG Dataset structure as specified by the EEG_DATASET parameters

    EEG_DATASET (dict):
        'Study' (str): the name of the study
        'Subjects' (list): list of subject names
        'States' (list): list of state names
        'Targets' (str of list): string or list with Target names
        'Target Values' (str or list): 'Read_from_info_file' or list of Targets

    Args:
        raw_eeg_dataset (dict): Structure including 'EEG Dataset Parameters', 'Subjects', 'EEG Data' and 'Info'
        EEG_DATASET (dict): the selected EEG dataset description
        id: an ID value of the dataset
    Returns:
        eeg_dataset, (Dict): Structure including 'EEG Dataset Parameters', 'Subjects', 'Dataset ID', 'EEG Data',
        'States', 'Channels', 'Targets', and 'Target Values'
    """

    subjects = []
    dataset_id = []
    eeg_data_list = []
    states_list = []
    channels_list = []
    targets_list = []

    for i, subject in enumerate(raw_eeg_dataset['Subjects']): # For each Subject
        if subject not in EEG_DATASET['Subjects']: # Select Subjects
            continue

        subjects.append(subject)
        dataset_id.append(id)
        channels_list.append(raw_eeg_dataset['Info'][i]['Channels'])

        eeg_data = np.empty((len(EEG_DATASET['States'])), dtype=object)
        states = []

        state_index = 0
        for s, state in enumerate(raw_eeg_dataset['Info'][i]['States']):  # For each State
            if state not in EEG_DATASET['States']: # Select States
                continue

            eeg_data[state_index] = raw_eeg_dataset['EEG Data'][i][s]
            states.append(state)
            state_index += 1

        eeg_data_list.append(eeg_data)
        states_list.append(states)

        if EEG_DATASET['Target Values'] is not None: # Check for Target Values

            targets = np.empty((len(EEG_DATASET['States'])), dtype=object)

            state_index = 0
            for s, state in enumerate(raw_eeg_dataset['Info'][i]['States']):  # For each State
                if state not in states:  # Select States
                    continue

                if EEG_DATASET['Target Values'] == 'Read_from_info_file':
                    if EEG_DATASET['Targets'] == '1-hot': # 1-hot
                        target = np.empty((len(eeg_data[state_index]), len(states)))
                        target[:] = [1 if i == state_index else 0 for i in range(len(states))]

                    else: # List of Target Keys
                        target = np.empty((len(eeg_data[state_index]), len(EEG_DATASET['Targets'])))
                        target[:] = [raw_eeg_dataset['Info'][i]['Export'][target_key][s]
                                      for target_key in EEG_DATASET['Targets']]

                else:  # List Provided
                    target = np.empty((len(eeg_data[state_index]), len(EEG_DATASET['Target Values'][state_index])))
                    target[:] = EEG_DATASET['Target Values'][state_index]

                targets[state_index] = target
                state_index += 1

            targets_list.append(targets)

        else: # No Target Values
            targets_list = None

    if targets_list is not None:
        targets_list = np.array(targets_list)

    eeg_dataset = {'EEG Dataset Parameters': raw_eeg_dataset['EEG Dataset Parameters'], 'Subjects': np.array(subjects),
                   'Dataset ID': np.array(dataset_id), 'EEG Data': np.array(eeg_data_list),
                   'States': np.array(states_list), 'Channels': np.array(channels_list),
                   'Targets': EEG_DATASET['Targets'], 'Target Values': targets_list}
    return eeg_dataset


def check_datasets_consistency(eeg_datasets):
    """
    Returns True, if the EEG datasets contained in eeg_datasets list are consistent with respect to:
    1) Sampling Frequency
    2) Reference Montage
    3) Channel Names
    4) EEG Representation
    5) Epoch Size
    6) Targets

    Args:
        eeg_datasets (list): list of eeg_datasets
    """

    # Check Sampling Frequency Consistency
    if len(set([eeg_dataset['EEG Dataset Parameters']['EEG_PARAMETERS']['Sfreq'] for eeg_dataset in eeg_datasets]))\
            > 1:
        print('Inconsitent Dataset Sampling Frequency')
        return False

    # Check Reference Montage
    if len(set([eeg_dataset['EEG Dataset Parameters']['EEG_PARAMETERS']['Reference'] for eeg_dataset in eeg_datasets]))\
            > 1:
        print('Inconsitent Dataset Reference Montage')
        return False

    # Check Channel Consistency
    channels_list = []
    for eeg_dataset in eeg_datasets:
        channels_list += [ch_list for ch_list in eeg_dataset['Channels']]
    channels_list = np.array(channels_list) # Get Channels from All Datasets and Subjects

    for channel_index in range(channels_list.shape[1]):  # For All Channel Positions
        if not _channel_name_consistency(channels_list[:, channel_index]):
            print('Inconsitent Dataset EEG Channels')
            return False

    # Check EEG Representation Consistency (Topomap)
    if all(['Topomap' in eeg_dataset['EEG Dataset Parameters']['EEG_PARAMETERS'] for eeg_dataset in eeg_datasets]):
        if len(set([eeg_dataset['EEG Dataset Parameters']['EEG_PARAMETERS']['Topomap'] for eeg_dataset in eeg_datasets])) \
                > 1:
            print('Inconsitent Dataset EEG Representation')
            return False

    # Check Epoch Size Consistency
    if len(set([eeg_dataset['EEG Dataset Parameters']['EEG_PARAMETERS']['Epoch Size'] for eeg_dataset in eeg_datasets])) \
            > 1:
        print('Inconsitent Dataset Epoch Size')
        return False

    # Check Target Consitency
    if len(set([tuple(eeg_dataset['Targets']) for eeg_dataset in eeg_datasets])) > 1:
        print('Inconsitent Dataset Targets')
        return False

    return True

def _channel_name_consistency(channels):
    """
    Returns True, if channel names in channels list are the same or consistent
    e.g. T3 can replace T7, T5 can replace P7, etc.
    """
    if len(set(channels)) == 1: # One channel name
        return True
    if set(channels) == {'T3', 'T7'}:
        return True
    if set(channels) == {'T5', 'P7'}:
        return True
    if set(channels) == {'T4', 'T8'}:
        return True
    if set(channels) == {'T6', 'P8'}:
        return True

    return False


def concatenate_datasets(eeg_datasets):
    """
    Concatenates eeg_datasets structures ('EEG Dataset Parameters', 'Subjects', 'Dataset ID', 'EEG Data',
        'States', 'Channels', 'Targets', and 'Target Values')

    Args:
        eeg_datasets (list): List of eeg_dataset structures
    Returns:
        eeg_dataset (Dict): The concatenated eeg_dataset structure
    """

    eeg_dataset = {}
    eeg_dataset['EEG Dataset Parameters'] = [eeg_dataset['EEG Dataset Parameters'] for eeg_dataset in eeg_datasets]

    for key in ['Subjects', 'Dataset ID', 'EEG Data', 'States', 'Channels']:
        eeg_dataset[key] = np.concatenate([eeg_dataset[key] for eeg_dataset in eeg_datasets])

    # Check for Consistent Dataset Targets
    if len(set([tuple(eeg_dataset['Targets']) for eeg_dataset in eeg_datasets])) > 1:
        print('Inconsitent Dataset Targets!')
        eeg_dataset['Targets'] = None
    else:
        eeg_dataset['Targets'] = eeg_datasets[0]['Targets']

    # Check for Target Values in All Datasets
    eeg_dataset['Target Values'] = None
    if all([eeg_dataset['Target Values'] is not None for eeg_dataset in eeg_datasets]):
        eeg_dataset['Target Values'] = np.concatenate([eeg_dataset['Target Values'] for eeg_dataset in eeg_datasets])
    else:
        print('Target values are missing in one or more datasets!')

    return eeg_dataset


# -------------------------------------------- EEG PRE-PROCESSING ------------------------------------------------------

def check_EEG_model_compatibility(eeg_data_list, model_name):
    """
    Returns True, if the data in eeg_data_list are compatible with the model specified in model_name

    Args:
        eeg_data_list (list): list of eeg data (subject,) (state,) (epoch, channel, sample)
        model_name (str): name of the Model
    """
    if model_name in ['Toy', 'cNN_3D', 'BD_Deep4', 'BD_Shallow', 'EEGNet_v1', 'EEGNet_v2']:
        if not all([len(eeg_data[0].shape)==3 for eeg_data in eeg_data_list]):  # (epoch, channel, sample)
            return False

    if model_name in ['cNN_topomap']:
        if not all([len(eeg_data[0].shape)==4 for eeg_data in eeg_data_list]):  # (epoch, channel, channel, sample)
            return False

    return True


def normalize_EEG_data_list(eeg_data_list, normalization_method='epoch_stand_robust'):
    """
    Performs EEG Normalization to EEG Data List
    Args:
        eeg_data_list (list): list of eeg data (subject,) (state,) (epoch, channel, sample)
        normalization_method (str): name of the Normalization method
    Returns:
        normalized_eeg_data_list (list): list of the normalized eeg data
    """
    normalized_eeg_data_list = np.empty((len(eeg_data_list),), dtype=object)

    for i, eeg_data in enumerate(eeg_data_list):  # For each Subject
        normalized_eeg_data = np.empty((len(eeg_data),), dtype=object)

        for s, X in enumerate(eeg_data):  # For each State
            # EEG Normalization
            normalized_eeg_data[s] = normalize_EEG_data(np.copy(X), normalization_method=normalization_method)

        normalized_eeg_data_list[i] = normalized_eeg_data

    return normalized_eeg_data_list


def normalize_EEG_data(X, normalization_method='epoch_stand_robust'):
    """
    Performs EEG Normalization to EEG Data
    This is a requirement for many machine learning models, as EEG data are typically in μV

    Normalization Methods:
        'epoch_stand': Epoch-wise standardization (mean=0, var=1)
        'epoch_stand_robust': Epoch-wise Robust Standardization (quartile range: 0.25 - 0.75)
        'epoch_l2_norm': Epoch-wise L2 Normalization

    Args:
        X (ndarray): Array of (epoch, channel, sample) or (epoch, channel, channel, sample)
        normalization_method (str): name of the Normalization method
    Returns:
        X (ndarray): X Normalized
    """

    # Flatten EEG epochs into 1D feature vectors
    X, original_shape = _flatten_X(X)

    if normalization_method == 'epoch_stand':
        X = preprocessing.scale(X, axis=1)

    elif normalization_method == 'epoch_stand_robust':
        X = preprocessing.robust_scale(X, axis=1)

    elif normalization_method == 'epoch_l2_norm':
        X = preprocessing.normalize(X, norm='l2')

    else:
        print('No valid normalization method')

    X = X.reshape(original_shape)
    return X

def _flatten_X(X):
    original_shape = X.shape
    if len(X.shape) == 3:  # (epochs, channels, samples)
        X = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2]))
    elif len(X.shape) == 4:  # (epochs, channels, channels, samples)
        X = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))

    return X, original_shape



def reshape_EEG_data_list(eeg_data_list, channels_list, model_name):
    """
    Reshapes EEG data based on Model Specifications

    Args:
        eeg_data_list (list): list of eeg data (subject,) (state,) (epoch, channel, sample)
        channels_list (list): list of channel names per subject
        model_name (str): name of the Model
    Returns:
        reshaped_eeg_data_list (list): the reshaped list of eeg data
    """
    reshaped_eeg_data_list = np.empty((len(eeg_data_list),), dtype=object)

    for i, eeg_data in enumerate(eeg_data_list):  # For each Subject
        reshaped_eeg_data = np.empty((len(eeg_data),), dtype=object)

        for s, X in enumerate(eeg_data):  # For each State
            # EEG Reshape
            reshaped_eeg_data[s] = reshape_EEG_data(np.copy(X), channels_list[i], model_name)

        reshaped_eeg_data_list[i] = reshaped_eeg_data

    return reshaped_eeg_data_list


def reshape_EEG_data(X, channels, model_name):
    """
    Reshapes EEG data based on Model Specifications
    e.g. For 1D models, remove the channel dimension
    e.g. For 3D models, expand the 1D channel dimension into a 2D structure (channel x channel)
    For 2D and 3D models, a kernel dimension is added as the last dimension within an instance

    Args:
        X (ndarray): Array of (epoch, channel, sample) or (epoch, channel, channel, sample)
        channels (list): list of channel names
        model_name (str): name of the Model
    Returns:
       X (list): the reshaped X
    """

    # Select Model
    if model_name in ['Toy']:  # 1D Models
        X = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2])) # Remove Channel Dimension

    elif model_name in ['cNN_3D']:  # cNN_3D Model

        if all([channel in EEG_10_20_compatible for channel in channels]):  # Check 10-20 Channel Compatibility
            X = _create_2D_grid(X, channels) # Create 2D Grid Representation
            X = np.expand_dims(X, axis=-1) # Add kernel
        else:
            print('Incompatible EEG Channels with 10-20 system')

    else: # 2D Models ('BD_Deep4', 'BD_Shallow', 'EEGNet_v1', 'EEGNet_v2') and 'cNN_topomap'
        X = np.expand_dims(X, axis=-1)  # Add kernel

    return X

def _create_2D_grid(X, channels):
    # Creates an X data array from (epochs, channels, samples) to (epochs, channels, channels, samples)
    # based on a 2D (5x5) Grid for 10-20 system

    grid_X = np.zeros((X.shape[0], 5, 5, X.shape[2]))
    for c, channel in enumerate(channels):
        (i, j) = EEG_2D_grid_map[channel]
        grid_X[:, i, j, :] = X[:, c, :]

    return grid_X


# ------------------------------------------------ EEG SAMPLE WEIGHTING ------------------------------------------------

def get_sample_weights_list(eeg_data_list, dataset_id=None, dataset_categories=None, W1=True, W2=True, W3=True, W4=True,
                            W5=True):
    """
    Extracts a sample weight list from the eeg_data_list data.
    This is important when training models with unbalanced or heterogeneous datasets, in terms of number of states,
    subjects, datasets, and other categorical differences

    Args:
        eeg_data_list (list): list of eeg data (subject,) (state,) (epoch, channel, sample)
        dataset_id (list): dataset ID for each subject
        dataset_categories (list): list of categories for each dataset ID
        W1 (boolean): State normalization
        W2 (boolean): Subject normalization
        W3 (boolean): Dataset normalization
        W4 (boolean): Category Normalization
        W5 (boolean): Full Normalization
    Returns:
       sample_weights_list (list): the list of weights per sample instance (subject,) (state,) (epoch, weight)
    """

    sample_weights_list = np.empty((len(eeg_data_list)), dtype=object)
    w1 = w2 = w3 = w4 = w5 = 1

    for i, eeg_data in enumerate(eeg_data_list):  # For each Subject
        state_weights = np.empty((len(eeg_data)), dtype=object)

        for s, X in enumerate(eeg_data):  # For each State

            sample_weights = np.empty((len(X), 1))

            if W1:
                w1 = 1 / len(X)  # no of epochs
            if W2:
                w2 = 1 / len(eeg_data)  # no of states
            if W3:
                if dataset_id is None:
                    w3 = 1 / len(eeg_data_list)  # no of subjects
                else:
                    w3 = 1 / list(dataset_id).count(dataset_id[i])  # no of subjects in current dataset

            if dataset_categories is not None:
                if W4:
                    w4 = 1 / list(dataset_categories).count(
                        dataset_categories[dataset_id[i]])  # no of datasets in current category
                if W5:
                    w5 = 1 / len(set(dataset_categories))  # no of categories

            sample_weights[:] = w1 * w2 * w3 * w4 * w5

            state_weights[s] = sample_weights
        sample_weights_list[i] = state_weights

    return sample_weights_list
