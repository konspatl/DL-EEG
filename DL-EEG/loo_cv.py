# Author: Konstantinos Patlatzoglou <konspatl@gmail.com>

# A script on Training and Testing deep learning models for state-based EEG decoding,
# using a leave-one-participant-out cross-validation approach.

import os, shutil
from pathlib import Path

import numpy as np
import json
from sklearn.model_selection import LeaveOneOut
from sklearn.utils import shuffle
from tensorflow.keras import backend as K

from utils import utils
from utils import EEG
from utils import NN


# --------------------------------------LOO-CV PARAMETERS--------------------------------------#
RESULTS_DIR_NAME = 'LOO-CV'  # Name of LOO-CV Export Directory


NO_OF_DATASETS = 1

EEG_DATASET_1 = {'Study': 'Cambridge Anesthesia',
                 'Subjects': ['S1', 'S2', 'S3'],
                 'States': ['Wakefulness', 'Sedation'],
                 'Targets': '1-hot', # '1-hot' (Classifcation), list [target1, target2, ] (Regression)
                 'Target Values': [[1,0], [0,1]], # None, 'Read_from_info_file', list [[], [], ...]
                 }

# EEG_DATASET_2 = {'Study': 'Liege Anesthesia',
#                  'Subjects': ['S4', 'S5', 'S6'],
#                  'States': ['Wakefulness', 'Sedation'],
#                  'Targets': '1-hot', # '1-hot' (Classifcation), list [target1, target2, ] (Regression)
#                  'Target Values': 'Read_from_info_file', # None, 'Read_from_info_file', list [[], [], ...]
#                  }

CLASSES = ['Wakefulness', 'Sedation']  # list (Classification) or None (Regression)

MODEL_PARAMETERS = {'Model': 'EEGNet_v2', # 'Toy', 'cNN_3D', 'cNN_topomap' 'BD_Deep4', 'BD_Shallow', 'EEGNet_v1', 'EEGNet_v2'
                    'EEG Normalization': 'epoch_stand_robust',  # 'epoch_stand', 'epoch_stand_robust', 'epoch_l2_norm'
                    'Sample Weights': True, #
                    'Target Weights': True,
                    'Shuffle Samples': True,
                    'Optimizer': 'Adadelta', # tensorflow  parameter ('Adadelta', 'Adam', 'SGD')
                    'Learning Rate': 1,
                    'Loss': 'categorical_crossentropy', # tensorflow parameter ('categorical_crossentropy', 'mean_squared_error')
                    'Metrics': ['accuracy'],  # tensorflow parameter ('accuracy', 'mean_absolute_error')
                    'Batch Size': 64,  # samples per gradient update
                    'Epochs': 10  # no of iterations over the training dataset
                    }

SAVE_MODELS = False  # Save Trained Model for Each Subject Independently
SEPARATE_DATASET_RESULTS = False  # Export Results in Separate Directories for each Dataset
# ---------------------------------------------------------------------------------------------


def main():

    current_path = Path.cwd()

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

    # ------------------------------- MODEL SELECTION -------------------------------
    print('Creating Model...')

    if targets_list is None:
        print('Cannot perform LOPOCV. Missing Targets')
        exit(4)

    # Model Info
    input_shape = eeg_data_list[0][0].shape[1:]  # Model Input Shape
    output_shape = targets_list[0][0].shape[1:]  # Model Output Shape
    sfreq = eeg_dataset_parameters[0]['EEG_PARAMETERS']['Sfreq']  # EEG Sampling Frequency

    if targets == '1-hot':  # Classification
        classification = True
    else:  # Regression
        classification = False

    # Create Model
    model = NN.create_model(MODEL_PARAMETERS['Model'], input_shape, sfreq, output_shape, classification=classification)
    model.summary()

    model_memory_usage = NN.get_model_memory_usage(model, MODEL_PARAMETERS['Batch Size'])
    print('Model Memory Usage:' + str(model_memory_usage) + ' GB')
    print()

    # Compile Model
    optimizer = NN.get_optimizer(MODEL_PARAMETERS['Optimizer'], learning_rate=MODEL_PARAMETERS['Learning Rate'])
    model.compile(optimizer=optimizer,
                  loss=MODEL_PARAMETERS['Loss'],
                  metrics=MODEL_PARAMETERS['Metrics'])

    model_info = {'Input shape': input_shape,
                  'Sfreq': sfreq,
                  'Output shape': output_shape,
                  'Targets': targets
                  }
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

    # Save Initial Weights
    model.save_weights(str(result_path / 'Model' / 'initial_weights.tf'), save_format='tf')
    # -------------------------------------------------------------------------------

    # ------------------------------MODEL TRAINING/TESTING---------------------------
    print('Performing Leave-One-Participant-Out Cross-Validation...')
    print('Subjects' + str(subjects))

    loo = LeaveOneOut()  # LEAVE-ONE-SUBJECT-OUT CROSS-VALIDATION

    for train_indices, test_index in loo.split(subjects):  # For each test subject

        test_subject = subjects[test_index]
        print('Test Subject: ' + str(test_subject))

        # ---------------------- Select Training/Testing Data -----------------------
        train_eeg_data_list = eeg_data_list[train_indices]
        train_targets_list = targets_list[train_indices]
        train_dataset_id = dataset_id[train_indices]

        test_eeg_data_list = eeg_data_list[test_index]
        test_targets_list = targets_list[test_index]
        test_dataset_id = dataset_id[test_index]

        # Sample Weights
        train_sample_weight_list = None
        if MODEL_PARAMETERS['Sample Weights']:

            dataset_categories = None
            if all(['Other' in eeg_dataset['EEG_DATASET'] for eeg_dataset in eeg_dataset_parameters]):
                dataset_categories = [eeg_dataset['EEG_DATASET']['Other']['Drug']
                                      for eeg_dataset in eeg_dataset_parameters]
            train_sample_weight_list = EEG.get_sample_weights_list(train_eeg_data_list, dataset_id=train_dataset_id,
                                                                   dataset_categories=dataset_categories)

        # Concatenate Subjects and States
        Xtr, Ytr, Xtr_w = utils.concatenate_data(train_eeg_data_list, train_targets_list,
                                                 sample_weights_list=train_sample_weight_list)
        Xte, Yte, _ = utils.concatenate_data(test_eeg_data_list, test_targets_list)

        # Target Weights
        if MODEL_PARAMETERS['Target Weights']:
            if MODEL_PARAMETERS['Sample Weights']:
                Xtr_w = Xtr_w * utils.get_target_weights(Ytr)
                Xtr_w *= len(Xtr_w) / np.sum(Xtr_w)
            else:
                Xtr_w = utils.get_target_weights(Ytr)

        # Shuffle Train Data
        if MODEL_PARAMETERS['Shuffle Samples']:
            if Xtr_w is None:
                Xtr, Ytr = shuffle(Xtr, Ytr)
            else:
                Xtr, Ytr, Xtr_w = shuffle(Xtr, Ytr, Xtr_w)

        del train_eeg_data_list, train_targets_list, train_sample_weight_list, test_eeg_data_list, test_targets_list

        # ------------------------- Model Fitting -----------------------------------
        print('Model Fitting...')
        print('Number of training instances: ' + str(Xtr.shape[0]))

        # Training Data
        Xtr_t, Ytr_t = utils.get_tensor_maps(Xtr, Y=Ytr, classification=classification)
        # Testing Data
        Xte_t, Yte_t = utils.get_tensor_maps(Xte, Y=Yte, classification=classification)

        # Initialize Weights
        model.load_weights(str(result_path / 'Model' / 'initial_weights.tf'))

        history = model.fit(Xtr_t, Ytr_t,
                            batch_size=MODEL_PARAMETERS['Batch Size'],
                            epochs=MODEL_PARAMETERS['Epochs'],
                            verbose=2,
                            validation_data=(Xte_t, Yte_t),
                            sample_weight=Xtr_w)

        history = history.history

        Ypred = model.predict(Xte_t, batch_size=MODEL_PARAMETERS['Batch Size'])
        if isinstance(Ypred, list): # If Multiple Regression Targets
            Ypred = utils.merge_targets(Ypred)

        score = model.evaluate(Xte_t, Yte_t, batch_size=MODEL_PARAMETERS['Batch Size'], verbose=0)
        score = dict(zip(model.metrics_names, score))

        # --------------------- Export Subject Results ------------------------------
        print('Export subject results...')

        # Select Dataset Export Path
        if SEPARATE_DATASET_RESULTS:
            dataset_export_path = result_path / ('EEG DATASET ' + str(test_dataset_id[0]+1))
        else:
            dataset_export_path = result_path / 'EEG DATASET'

        if not dataset_export_path.exists():
            utils.create_directory(dataset_export_path)

        # Save Model
        if SAVE_MODELS:
            model.save(str(dataset_export_path / (test_subject[0] + '_model.tf')), include_optimizer=True,
                       save_format='tf')

        # Calculate per-state Predictions and Target Values
        state_epochs = [state.shape[0] for state in eeg_data_list[test_index[0]]]
        predictions = utils.reshape_predictions(Ypred, state_epochs)
        target_values = [list(target[0]) for target in targets_list[test_index[0]]]

        # Subject Info
        subject_info = {'States': list(states_list[test_index[0]]),
                        'Channels': list(channels_list[test_index[0]]),
                        'Target Values': target_values,
                        'Score': score,
                        'History': history
                        }

        # Export Predictions as numpy array
        np.save(str(dataset_export_path / (test_subject[0] + '_predictions.npy')), predictions, allow_pickle=True)
        # Export Subject Info as a json file
        json.dump(subject_info, open(str(dataset_export_path / (test_subject[0] + '_info.json')), 'w'), indent=4)
        # ---------------------------------------------------------------------------

        K.clear_session()  # Reset Tensorflow State (Clears Model Memory)
        print()
    # -------------------------------------------------------------------------------

    # Delete Model Weights
    shutil.rmtree(result_path / 'Model')


if __name__ == '__main__':
    main()
