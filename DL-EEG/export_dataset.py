# Author: Konstantinos Patlatzoglou <konspatl@gmail.com>

import numpy as np
import json
from pathlib import Path

from datasetUtils import EEG
from datasetUtils import topomap
from datasetUtils import utils


# ---------------------------------------EXPORT PARAMETERS-----------------------------------
DATASET_NAME = 'Anesthesia Dataset'  # Name of Dataset Directory
EXPORT_NAME = 'Anesthesia Export' # Name of Export Directory


EEG_DATASET = {'Study': 'Cambridge Anesthesia',
               'Subjects': ['S1', 'S2', 'S3'],
               'States': ['Wakefulness', 'Sedation'],
               'Export': ['Drug Levels'],  # list (export keys) (Optional)
               'Other':  # Dict (Optional)
                   {'Drug': 'Propofol'}
               }

EEG_PARAMETERS = {'Montage': 'GSN-HydroCel-129',  # 'GSN-HydroCel-129' (mne argument)
                  'Channels': '10-20',  # 'All', '10-20'
                  'Crop Time': None,  # [tmin, tmax] (sec)
                  'Filtering': [0.5, 40],  # [Low-cut freq, High-cut freq]
                  'Notch Freq': None,  # 50/60 Hz (Notch filter frequency, including harmonics)
                  'Sfreq': 100,  # Hz
                  'Epoch Size': 1,  # (sec)
                  'Epoch Overlap': 0,  # % overlap (0%-99%)
                  'Epoch Baseline Correction': False,
                  'Epoch Peak-to-Peak Threshold': 800e-6,  # peak-to-peak amplitude threshold (V) or None
                  'Interpolate Bad Channels': True, # Bad if signal is flat, or If 20% of the signal exceeds p-t-p threshold
                  'Epoch Rejection': True,  # Reject epoch if p-t-p threshold exceeds in more than 20% of the channels
                  'Reference': 'Average',  # 'Default' (Cz), 'Average', 'Cz', 'Frontal' (Fp1-F3-Fz-F4-Fp2)
                  'Exponential Moving Standardization': False,  # Performs EMS on EEG data - (Optional)
                  'Topomap': False  # Extracts Topomap Representation for EEG Epochs - (Optional)
                  }
# -------------------------------------------------------------------------------------------


def main():

    current_path = Path.cwd()

    # --------------------------- IMPORT DATASET INFO -------------------------------
    print('Importing Dataset Info from dataset_info.json file...')

    dataset_path = current_path.parent / 'data' / DATASET_NAME

    # Import Dataset Info from 'dataset_info.json' file
    dataset_info = None
    try:
        with open(str(dataset_path / 'dataset_info.json')) as json_file:
            dataset_info = json.load(json_file)
    except FileNotFoundError:
        print('dataset_info.json file is missing!')
        exit(1)

    if not utils.check_dataset_consistency(dataset_info):
        print('Inconsistent dataset info')
        exit(1)
    # -------------------------------------------------------------------------------

    # dataset_info (dict):
    #   'Subjects' (list): list of Subject names
    #   'States' (list): list of States per subject
    #   'EEG Files' (list): list of EEG files per subject
    #   'Export' (Dict): dict of export values - (Optional)
    #   'Montage File' (str): path to EEG montage file - (Optional)

    subjects = dataset_info['Subjects']
    states = dataset_info['States']
    EEG_files = dataset_info['EEG Files']
    export = None
    montage_file = None

    if 'Export' in dataset_info:
        export = dataset_info['Export']
    if 'Montage File' in dataset_info:
        montage_file = Path(dataset_info['Montage File'])
    # -------------------------------------------------------------------------------

    # ------------------------- CREATE EXPORT PATH DIR ------------------------------
    print('Creating Export Directory...')

    # Create 'Export Data' Directory
    export_path = dataset_path.parent / EXPORT_NAME
    utils.create_directory(export_path)

    # Create JSON file with EEG Dataset EXPORT PARAMETERS
    EEG_DATASET_PARAMETERS = {'EEG_DATASET': EEG_DATASET, 'EEG_PARAMETERS': EEG_PARAMETERS}
    json.dump(EEG_DATASET_PARAMETERS, open(str(export_path / 'EEG_DATASET_PARAMETERS.json'), 'w'), indent=4)
    # -------------------------------------------------------------------------------

    # --------------------------- EXPORT EEG DATASET --------------------------------
    print('Exporting EEG Dataset...')

    # For each Selected Subject
    for i, subject in enumerate(subjects):
        if subject not in EEG_DATASET['Subjects']: # Filter Subject Selection
            continue

        # Subject Data and Info
        eeg_data = np.empty((len(EEG_DATASET['States']),), dtype=object)
        selected_states = []
        channels = []
        selected_export = None
        if 'Export' in EEG_DATASET: # Check for Export
            selected_export = {key: [] for key in EEG_DATASET['Export']}

        # For each Selected State
        for s, state in enumerate(states[i]):
            if state not in EEG_DATASET['States']: # Filter State Selection
                continue

            # Get EEG File and Montage File (Optional)
            EEG_file_path = dataset_path / EEG_files[i][s]

            if montage_file is not None:
                montage_file = dataset_path / montage_file

            # MNE RAW Object
            original_raw, montage = EEG.load_EEG_data(EEG_file_path, EEG_PARAMETERS['Montage'],
                                                      montage_file=montage_file)

            # MNE Preprocessed RAW and EPOCHS
            preprocessed_raw, epochs = EEG.process_EEG_data(original_raw.copy(), EEG_PARAMETERS)

            # Numpy Array
            epoched_data = epochs.get_data()

            if 'Exponential Moving Standardization' in EEG_PARAMETERS:
                if EEG_PARAMETERS['Exponential Moving Standardization']:
                    epoched_data = EEG.exponential_moving_standardization(np.copy(epoched_data), EEG_PARAMETERS['Sfreq'])

            if 'Topomap' in EEG_PARAMETERS:
                if EEG_PARAMETERS['Topomap']:
                    epoched_data = topomap.create_topomap(np.copy(epoched_data), EEG_PARAMETERS['Sfreq'],
                                                          epochs.ch_names, montage, pixel_res=30)

            eeg_data[s] = epoched_data
            selected_states.append(state)
            channels = epochs.ch_names
            if 'Export' in dataset_info and 'Export' in EEG_DATASET: # Check for Export
                for key in export.keys():
                    if key in EEG_DATASET['Export']:
                        selected_export[key].append(export[key][i][s])

        # Subject Info
        subject_info = {'States': selected_states,
                        'Channels': channels,
                        'Export': selected_export
                        }

        # Export EEG Data as numpy array
        np.save(str(export_path / (subject + '_eeg_data.npy')), eeg_data, allow_pickle=True)
        # Export Subject Info as a json file
        json.dump(subject_info, open(str(export_path / (subject + '_info.json')), 'w'), indent=4)

    # -------------------------------------------------------------------------------


if __name__ == '__main__':
    main()
