# Author: Konstantinos Patlatzoglou <konspatl@gmail.com>

# A script on Testing deep learning models for state-based EEG decoding.

import os
from pathlib import Path

import numpy as np
import json

from utils import utils
from utils import EEG
from utils import NN


# -----------------------------------MODEL TESTING PARAMETERS----------------------------------
MODEL_DIR_NAME = 'Model Training'  # Name of Model Training Directory
RESULTS_DIR_NAME = 'Model Testing'  # Name of Model Testing Directory


NO_OF_DATASETS = 1

EEG_DATASET_1 = {'Study': 'Cambridge Anesthesia',
                 'Subjects': ['S3'],
                 'States': ['Wakefulness', 'Sedation'],
                 'Targets': ['Drug Levels'], # '1-hot' (Classifcation), list [target1, target2, ] (Regression)
                 'Target Values': None, # None, 'Read_from_info_file', list [[], [], ...]
                 }

SEPARATE_DATASET_RESULTS = False  # Export Results in Separate Directories for each Dataset
# ---------------------------------------------------------------------------------------------


def main():

    current_path = Path.cwd()

    # --------------------------------- LOAD MODEL ----------------------------------
    print('Loading Model...')

    model_path = current_path.parent / 'results' / MODEL_DIR_NAME

    # Load MODEL_PARAMETERS and MODEL_INFO Structures
    MODEL_PARAMETERS = json.load(open(str(model_path / 'MODEL_PARAMETERS.json')))
    model_info = json.load(open(str(model_path / 'MODEL_INFO.json')))

    CLASSES = None
    if model_info['Targets'] == '1-hot':
        CLASSES = json.load(open(str(model_path / 'CLASSES.json')))

    # Load Model
    model = NN.get_model(str(model_path / 'Model.tf'))
    model.summary()

    model_memory_usage = NN.get_model_memory_usage(model, MODEL_PARAMETERS['Batch Size'])
    print('Model Memory Usage:' + str(model_memory_usage) + ' GB')
    print()

    MODEL_CHECKPOINT = False
    checkpoint_models = None
    training_history = None

    # Check for Model Checkpoint (Load Epoch Models + Training history)
    if os.path.exists(model_path / 'Model Checkpoint'):
        MODEL_CHECKPOINT = True
        checkpoint_models = NN.get_checkpoint_models(model_path / 'Model Checkpoint')
        training_history = json.load(open(str(model_path / 'Model Checkpoint' / 'training_history.json')))
    # -------------------------------------------------------------------------------

    # ---------------------------- IMPORT EEG DATASETS ------------------------------
    print('Importing EEG Datasets...')

    datasets_path = current_path.parent / 'data'

    eeg_datasets = []
    for i in range(NO_OF_DATASETS): # For each specified Dataset

        EEG_DATASET = globals()['EEG_DATASET_' + str(i+1)]
        dataset_path = EEG.find_dataset_path(datasets_path, EEG_DATASET)

        if dataset_path is None:
            print('No dataset with the given parameters found')
            exit(1)

        # Get the Dataset from the corresponding directory
        # Dict ('EEG Dataset Parameters', 'Subjects', 'EEG Data', 'Info')
        raw_eeg_dataset = EEG.get_dataset(dataset_path)

        # Select the Data specified in EEG_DATASET with respect to Subjects, States and Targets
        # Dict ('EEG Dataset Parameters', 'Subjects', 'Dataset ID', 'EEG Data', 'States', 'Channels', 'Targets',
        #       'Target Values')
        eeg_dataset = EEG.select_data_from_dataset(raw_eeg_dataset, EEG_DATASET, id=i)
        eeg_datasets.append(eeg_dataset)

    # Check Datasets Consistency (Sampling Frequency, Reference Montage, Channel names, Epoch Size, Targets)
    print('Checking Dataset Consistency...')
    if not EEG.check_datasets_consistency(eeg_datasets):
        print('Datasets cannot be integrated')
        exit(2)

    # Concatenate All Datasets
    eeg_dataset = EEG.concatenate_datasets(eeg_datasets)
    del eeg_datasets
    # -------------------------------------------------------------------------------

    # ---------------------------- EEG DATA AND TARGETS -----------------------------

    # eeg_dataset_parameters (list): List of EEG_DATASET_PARAMETERS for each dataset
    # subjects (list): list of subjects (subject,)
    # dataset_id (list): list of dataset ids (subject,)
    # eeg_data_list (list): list of eeg data (subject,) (state,) (epoch, channel, sample)
    # states_list (list): list of states (subject,) (state,)
    # channels_list (list): list of channel names (subject,)
    # targets (str or list): target name or list of target names
    # targets_list (list or None): list of target values (subject,) (state,) (epoch, target)

    eeg_dataset_parameters = eeg_dataset['EEG Dataset Parameters']
    subjects = eeg_dataset['Subjects']
    dataset_id = eeg_dataset['Dataset ID']
    eeg_data_list = eeg_dataset['EEG Data']
    states_list = eeg_dataset['States']
    channels_list = eeg_dataset['Channels']
    targets = eeg_dataset['Targets']
    targets_list = eeg_dataset['Target Values']

    del eeg_dataset
    # -------------------------------------------------------------------------------

    # ------------------------------ EEG PRE-PROCESSING -----------------------------
    print('Performing EEG Data Pre-processing...')

    # Check Topomap Model Compatibility
    if MODEL_PARAMETERS['Model'] == 'cNN_topomap':
        if not EEG.check_EEG_model_compatibility(eeg_data_list, 'cNN_topomap'):
            print('EEG data shape is incompatible with selected Model')
            exit(3)

    # Perform EEG Normalization (e.g. epoch-wise Standardization)
    eeg_data_list = EEG.normalize_EEG_data_list(eeg_data_list, MODEL_PARAMETERS['EEG Normalization'])

    # Reshape EEG data according to Model + Add Kernel Dimension
    # e.g. If Toy Model (1D), reshape EEG data into (epoch, sample)
    # e.g. If cNN_3D model (3D), reshape EEG data into (epoch, channel, channel, sample, 1)
    eeg_data_list = EEG.reshape_EEG_data_list(eeg_data_list, channels_list, MODEL_PARAMETERS['Model'])
    # -------------------------------------------------------------------------------

    # --------------------------CHECK EEG DATA / MODEL CONSISTENCY-------------------
    print('Checking EEG Data and Model Consistency...')

    # Model Info
    input_shape = eeg_data_list[0][0].shape[1:]  # Model Input Shape
    sfreq = eeg_dataset_parameters[0]['EEG_PARAMETERS']['Sfreq']  # EEG Sampling Frequency

    if targets == '1-hot':  # Classification
        classification = True
    else:  # Regression
        classification = False

    # If Target Values
    if targets_list is None:
        output_shape = tuple(model_info['Output shape'])
    else:
        output_shape = targets_list[0][0].shape[1:]  # Model Output Shape

    # Check Model Consistency
    if input_shape != tuple(model_info['Input shape']) \
        or output_shape != tuple(model_info['Output shape']) \
        or sfreq != model_info['Sfreq'] \
        or targets != model_info['Targets']:

        print('EEG Data are inconsistent with selected model!')
        exit(4)
    # -------------------------------------------------------------------------------

    # ----------------------------- CREATE RESULTS DIRECTORY-------------------------
    print('Creating Result Directory...')

    result_path = current_path.parent / 'results' / RESULTS_DIR_NAME
    utils.create_directory(result_path)

    # For each Dataset, save EEG_DATASET_PARAMETERS (EEG_DATASET + EEG_PARAMETERS)
    for i in range(NO_OF_DATASETS):

        EEG_DATASET = globals()['EEG_DATASET_' + str(i + 1)]
        if 'Other' in eeg_dataset_parameters[i]['EEG_DATASET']:  # Concatenate 'Other' in EEG_DATASET
            EEG_DATASET['Other'] = eeg_dataset_parameters[i]['EEG_DATASET']['Other']
        EEG_PARAMETERS = eeg_dataset_parameters[i]['EEG_PARAMETERS']

        EEG_DATASET_PARAMETERS = {'EEG_DATASET': EEG_DATASET, 'EEG_PARAMETERS':EEG_PARAMETERS}
        json.dump(EEG_DATASET_PARAMETERS, open(str(result_path / ('EEG_DATASET_PARAMETERS_' + str(i + 1) + '.json')),
                                               'w'), indent=4)

    # Save MODEL_PARAMETERS
    json.dump(MODEL_PARAMETERS, open(str(result_path / 'MODEL_PARAMETERS.json'), 'w'), indent=4)

    # Save MODEL_INFO
    json.dump(model_info, open(str(result_path / 'MODEL_INFO.json'), 'w'), indent=4)

    # If Classification, Save CLASSSES
    if model_info['Targets'] == '1-hot':
        json.dump(CLASSES, open(str(result_path / 'CLASSES.json'), 'w'), indent=4)
    # -------------------------------------------------------------------------------

    # -----------------------------------MODEL TESTING-------------------------------
    print('Performing Model Testing...')
    print('Subjects' + str(subjects))

    for test_index, test_subject in enumerate(subjects):

        print('Test Subject: ' + test_subject)

        # Concatenate States
        Xte, Yte, _ = utils.concatenate_data(eeg_data_list, targets_list=targets_list)

        # Testing Data
        Xte_t, Yte_t = utils.get_tensor_maps(Xte, Y=Yte, classification=classification)

        # Model Prediction
        Ypred = model.predict(Xte_t, batch_size=MODEL_PARAMETERS['Batch Size'])
        if isinstance(Ypred, list): # If Multiple Regression Targets
            Ypred = utils.merge_targets(Ypred)

        score = None
        history = None

        # Model Evalution (Score)
        if Yte_t is not None:  # If Target Values
            score = model.evaluate(Xte_t, Yte_t, batch_size=MODEL_PARAMETERS['Batch Size'], verbose=0)
            score = dict(zip(model.metrics_names, score))

            # Model Checkpoint (History)
            if MODEL_CHECKPOINT:
                metrics_names = [('val_' + metric_name) for metric_name in model.metrics_names]
                history = []

                for epoch in range(MODEL_PARAMETERS['Epochs']): # For each Epoch
                    epoch_score = checkpoint_models[epoch].evaluate(Xte_t, Yte_t,
                                                                    batch_size=MODEL_PARAMETERS['Batch Size'],
                                                                    verbose=0)
                    history.append(epoch_score)
                history = [list(np.array(history)[:, metric]) for metric in range(len(model.metrics_names))]
                history = dict(zip(metrics_names, history))
                history.update(training_history) # Add Training History

        # --------------------- Export Subject Results ------------------------------
        print('Export Subject results...')

        # Select Subject Export Path
        if SEPARATE_DATASET_RESULTS:
            dataset_export_path = result_path / ('EEG DATASET ' + str(dataset_id[test_index]+1))
        else:
            dataset_export_path = result_path / 'EEG DATASET'

        if not dataset_export_path.exists():
            utils.create_directory(dataset_export_path)


        # Calculate per-state Predictions and Target Values
        state_epochs = [state.shape[0] for state in eeg_data_list[test_index]]
        predictions = utils.reshape_predictions(Ypred, state_epochs)
        target_values = None

        if targets_list is not None: # If Target Values
            target_values = [list(target[0]) for target in targets_list[test_index]]

        # Subject Info
        subject_info = {'States': list(states_list[test_index]),
                        'Channels': list(channels_list[test_index]),
                        'Target Values': target_values,
                        'Score': score,
                        'History': history
                        }

        # Export Predictions as numpy array
        np.save(str(dataset_export_path / (test_subject + '_predictions.npy')), predictions, allow_pickle=True)
        # Export Subject Info as a json file
        json.dump(subject_info, open(str(dataset_export_path / (test_subject + '_info.json')), 'w'), indent=4)
        # ---------------------------------------------------------------------------

        print()
    # -------------------------------------------------------------------------------



if __name__ == '__main__':
    main()