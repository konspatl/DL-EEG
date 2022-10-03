# Author: Konstantinos Patlatzoglou <konspatl@gmail.com>

import os, shutil
from pathlib import Path
from natsort import natsort_key

import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from scipy.signal import medfilt
from scipy.signal import convolve


# ------------------------------------------------GENERAL UTIL METHODS--------------------------------------------------

def create_directory(path):
    if (os.path.exists(path)):
        shutil.rmtree(path)
    os.makedirs(path)

def path_natural_sort(l):
    return sorted(l, key=lambda x: tuple(natsort_key(k) for k in Path(x).parts))

# ---------------------------------------------DATA PROCESSING METHODS--------------------------------------------------

def concatenate_data(eeg_data_list, targets_list=None, sample_weights_list=None):
    """
    Concatenates EEG data, targets (optional) and sample weights (optional) for all Subjects and States

    Args:
        eeg_data_list (list): list of eeg data (subject,) (state,) (epoch, channel, sample)
        targets_list (list): list of targets (subject,) (state,) (epoch, target) - (Optional)
        sample_weights_list (list): list of sample weights (subject,) (state,) (epoch, weight) - (Optional)
    Returns:
        X (ndarray): Array of Data (epoch, channel, sample)
        Y (ndarray): Array of Targets (epoch, target) - (Optional)
        X_w (ndarray): Array of Weights (epoch, weight) - (Optional)
    """

    X = _concatenate_states(np.copy(eeg_data_list))
    X = _concatenate_subjects(X)

    Y = None
    if targets_list is not None:
        Y = _concatenate_states(np.copy(targets_list))
        Y = _concatenate_subjects(Y)

    X_w = None
    if sample_weights_list is not None:
        X_w = _concatenate_states(np.copy(sample_weights_list))
        X_w = _concatenate_subjects(X_w)
        X_w *= len(X_w) / np.sum(X_w)  # Normalize to default summation

    return X, Y, X_w

def _concatenate_states(eeg_data_list):

    concat_eeg_data_list = np.empty((len(eeg_data_list),), dtype=object)
    for i, eeg_data in enumerate(eeg_data_list):  # For each subject
        concat_eeg_data_list[i] = np.vstack(tuple(eeg_data))

    return concat_eeg_data_list

def _concatenate_subjects(eeg_data_list):
    return np.vstack(tuple(eeg_data_list))


def get_target_weights(Y):
    """
    Extracts target weights for each target in Y, based on the number of instances per target

    Args:
        Y (ndarray): Array of Targets (epoch, target)
    Returns:
        Y_w (ndarray): Array of Target weights
    """
    Y_w = np.empty((len(Y), 1))

    # Find Unique Targets and No. of Instances per Unique Target
    unique_targets = np.unique(Y, axis=0)
    no_of_instances_per_target = [sum(np.all(Y == target, axis=-1)) for target in unique_targets]

    # Calculate Target weights
    for i, y in enumerate(Y): # For each target
        target_index = np.where(np.all(unique_targets == y, axis=-1))[0][0]
        Y_w[i] = 1 / (no_of_instances_per_target[target_index] * len(unique_targets))

    Y_w *= len(Y_w) / np.sum(Y_w)  # Normalize to default summation
    return Y_w


def get_tensor_maps(X, Y=None, classification=True):
    """
    Returns a dict mapping for the input/output names of a tensorflow model and the corresponding array/tensors

    Args:
        X (ndarray): the input numpy array
        Y (ndarray): the output numpy array
        classification (boolean): if True, one main_output, else (output_x for each Target key)
    Returns:
        X_t, Y_t (dict, dict): the dict mapping of X and Y
    """
    X_t = {'main_input': X}
    Y_t = None

    if Y is not None: # If Target Values
        if classification:  # Classification (main output)
            Y_t = {'main_output': Y}

        else:  # Regression (output_x for each target)
            Y_separated = separate_targets(Y)
            Y_t = {}
            for t, target in enumerate(Y_separated):
                Y_t['output_' + str(t)] = target

    return X_t, Y_t

def separate_targets(Y):
    """
    Separates Target Values into Indepedent Arrays
    i.e. from (instance, targets) to (targets,)(instance, 1)

    Args:
        Y (ndarray): Array of Targets (epoch, target)
    Returns:
        Y_separated (list): List of Separated Target Arrays
    """
    Y_separated = []

    for t in range(Y.shape[1]):
        Y_separated.append(np.expand_dims(Y[:, t], axis=-1))

    return Y_separated

def merge_targets(Y_separated):
    """
    Merges Target Values into a Single Array
    i.e. from (targets,)(instance, 1) to (instance, targets)

    Args:
        Y_separated (list): List of Separated Target Arrays
    Returns:
        Y (ndarray): Array of Targets (epoch, target)
    """
    return np.hstack(tuple(Y_separated))


def reshape_predictions(Ypred, state_epochs):
    """
    Reshapes model predictions (Ypred) into per-state predictions, based on the number of epochs per state

    Args:
        Ypred (ndarray): model predictions
        state_epochs (list): number of epochs per state
    Returns:
        predictions (ndarray): the reshaped predictions Array (state,)(epoch, target)
    """
    predictions = np.empty((len(state_epochs)), dtype=object)

    start_index = 0
    for s, epochs in enumerate(state_epochs): # For each State
        end_index = start_index + epochs
        predictions[s] = Ypred[start_index:end_index, :]
        start_index = end_index

    return predictions


def create_state_targets(state_epochs, target_values):
    """
    Creates a state_targets array given the number of instances per state and the target values

    Args:
        state_epochs (list): number of epochs per state
        target_values (list): the target values
    Returns:
        state_targets (ndarray): Array of Targets (state,) (epoch, target)
    """
    state_targets = np.empty((len(state_epochs)), dtype=object)

    for s, epochs in enumerate(state_epochs): # For each State
        state_targets[s] = np.empty((epochs, len(target_values[s])))
        state_targets[s][:] = target_values[s]

    return state_targets


def smoothen_predictions(predictions, method='Moving Average', kernel_size=3):
    """
    Returns predictions after applying a 'Moving Average' or 'Median' Filter

    Args:
        predictions (ndarray): the predictions Array (state,)(epoch, target)
        method (str): 'Moving Average', 'Median Filter'
        kernel_size (int): the kernel size of the filter
    Returns:
        smoothed_predictions (ndarray): the smoothened predictions Array
    """
    smoothened_predictions = np.empty((len(predictions)), dtype=object)

    for s, state_predictions in enumerate(predictions): # For each State

        state_smoothened_predictions = np.empty(state_predictions.shape)
        for t in range(state_predictions.shape[1]): # For each Target
            if method == 'Moving Average':
                state_smoothened_predictions[:, t] = convolve(state_predictions[:, t],
                                                            np.ones((kernel_size)) / kernel_size,
                                                            mode='same', method='direct')
            elif method == 'Median Filter':
                state_smoothened_predictions[:, t] = medfilt(state_predictions[:, t], kernel_size=kernel_size)

        smoothened_predictions[s] = state_smoothened_predictions

    return smoothened_predictions

# ----------------------------------------------------- STATISTICS -----------------------------------------------------

def confusionMatrix(predictions, target_values, no_of_classes):
    """
    Computes a confusion matrix based on the Targets and Model Predictions

    Args:
        predictions (ndarray): the predictions Array (state,)(epoch, target)
        target_values (list): the target values
        no_of_classes (int): the number of classes
    Returns:
        confusionMatrix (ndarray): the confusion matrix
    """
    # State Info
    state_epochs = [state.shape[0] for state in predictions]
    state_targets = create_state_targets(state_epochs, target_values)

    # Predictions and Targets
    Ypred = np.vstack(tuple(predictions))
    Yte = np.vstack(tuple(state_targets))

    # Transform from binary labels to multi-class labels
    lb = LabelBinarizer()
    lb.fit(np.arange(no_of_classes))
    Yte_mc = lb.inverse_transform(Yte)
    Ypred_mc = lb.inverse_transform(Ypred)
    labels = np.arange(no_of_classes)

    # Compute Confusion Matrix
    confusionMatrix = confusion_matrix(Yte_mc, Ypred_mc, labels=labels)
    return confusionMatrix

