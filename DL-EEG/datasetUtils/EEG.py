# Author: Konstantinos Patlatzoglou <konspatl@gmail.com>

import os
import warnings
import numpy as np
from natsort import natsorted
import pandas as pd
import mne

warnings.filterwarnings("ignore", category=DeprecationWarning)


# -------------------------------------------- EEG 10-20 SYSTEM CHANNELS------------------------------------------------
EEG_10_20 = ['F7', 'T3', 'T5', 'Fp1', 'F3', 'C3', 'P3', 'O1', 'Fz', 'Cz', 'Pz', 'Oz', 'Fp2', 'F4', 'C4', 'P4', 'O2',
             'F8', 'T4', 'T6']

# --------------------------------------------- EXIMIA 10-20 CHANNELS --------------------------------------------------
EXIMIA_10_20 = ['F7', 'T7', 'P7', 'Fp1', 'F3', 'C3', 'P3', 'O1', 'Fz', 'Cz', 'Pz', 'Oz', 'Fp2', 'F4', 'C4', 'P4', 'O2',
                'F8', 'T8', 'P8']

# -------------------------------------------EGI GSN HYDROCEL 256/257-----------------------------------------------
EGI257_10_20 = ['E2', 'E15', 'E19', 'E33', 'E41', 'E47', 'E59', 'E69', 'E86', 'E95', 'E101', 'E124', 'E137',
                'E149', 'E162', 'E178', 'E183', 'E202', 'E214', 'E257']

EGI257_10_20_map = {'E2': 'F8', 'E15': 'Fz', 'E19': 'Fp2', 'E33': 'Fp1', 'E41': 'F3', 'E47': 'F7', 'E59': 'C3',
                    'E69': 'T3', 'E86': 'P3', 'E95': 'T5', 'E101': 'Pz', 'E124': 'O1', 'E137': 'Oz', 'E149': 'O2',
                    'E162': 'P4', 'E178': 'T6', 'E183': 'C4', 'E202': 'T4', 'E214': 'F4', 'E257': 'Cz'}

EGI257_10_20_inv_map = {v: k for k, v in EGI257_10_20_map.items()}

# Exclude peripheral channels
EGI257_excl = ['E31', 'E67', 'E73', 'E82', 'E91', 'E92', 'E93', 'E94', 'E102', 'E103', 'E104', 'E105', 'E111', 'E112',
               'E113', 'E114', 'E120', 'E121', 'E122', 'E123', 'E133', 'E134', 'E135', 'E136', 'E145', 'E146', 'E147',
               'E148', 'E156', 'E157', 'E158', 'E165', 'E166', 'E167', 'E168', 'E174', 'E175', 'E176', 'E177', 'E187',
               'E188', 'E189', 'E190', 'E199', 'E200', 'E201', 'E208', 'E209', 'E216', 'E217', 'E218', 'E219', 'E225',
               'E226', 'E227', 'E228', 'E229', 'E230', 'E231', 'E232', 'E233', 'E234', 'E235', 'E236', 'E237', 'E238',
               'E239', 'E240', 'E241', 'E242', 'E243', 'E244', 'E245', 'E246', 'E247', 'E248', 'E249', 'E250', 'E251',
               'E252', 'E253', 'E254', 'E255', 'E256']

# ---------------------------------------------EGI GSN HYDROCEL 128/129-------------------------------------------------
EGI129_10_20 = ['E9', 'E11', 'E22', 'E24', 'E33', 'E36', 'E45', 'E52', 'E58', 'E62', 'E70', 'E75', 'E83', 'E92',
                'E96', 'E104', 'E108', 'E122', 'E124', 'E129']

EGI129_10_20_map = {'E9': 'Fp2', 'E11': 'Fz', 'E22': 'Fp1', 'E24': 'F3', 'E33': 'F7', 'E36': 'C3', 'E45': 'T3',
                    'E52': 'P3', 'E58': 'T5', 'E62': 'Pz', 'E70': 'O1', 'E75': 'Oz', 'E83': 'O2', 'E92': 'P4',
                    'E96': 'T6', 'E104': 'C4', 'E108': 'T4', 'E122': 'F8', 'E124': 'F4', 'E129': 'Cz'}

EGI129_10_20_inv_map = {v: k for k, v in EGI129_10_20_map.items()}

# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------MNE PROCESSING----------------------------------------------------------

def load_EEG_data(EEG_file, montage_name=None, montage_file=None):
    """
    MNE Read file and montage

    Args:
        EEG_file, (str): EEG filename
        montage, (String): name of montage (optional)
        montage_file, (Path): an mne compatible montage file (optional)

    Returns:
        raw, (MNE structure) - the MNE raw structure
        montage, (MNE structure) - the montage
    """

    raw = None
    montage = None

    file_extension = os.path.splitext(EEG_file)[1]

    if (file_extension == '.raw'):  # EGI format
        raw = mne.io.read_raw_egi(EEG_file, preload=True)

    elif (file_extension == '.set'):  # EEGLAB format
        try:
            raw = mne.io.read_raw_eeglab(EEG_file, preload=True)
        except:
            print('EEGLAB file contains epoched Data!')
            epochs = mne.io.read_epochs_eeglab(EEG_file)
            info = mne.create_info(epochs.info['ch_names'], epochs.info['sfreq'], ch_types='eeg')
            raw_data = np.hstack(tuple(epochs.get_data()))
            raw = mne.io.RawArray(raw_data, info)

    elif (file_extension == '.nxe'):  # EXIMIA format
        raw = mne.io.read_raw_eximia(EEG_file, preload=True)
        raw.apply_function(lambda x: x * 1e-6, picks=['eeg'])  # Eximia is shown in μV in mne. Transform to V.

    elif (file_extension == '.fif'):  # MNE format
        raw = mne.io.read_raw_fif(EEG_file, preload=True)

    # If Montage Name is given...
    if montage_name is not None:

        if montage_name in ['GSN-HydroCel-128', 'GSN-HydroCel-129', 'GSN-HydroCel-256', 'GSN-HydroCel-257']:
            # Fix EGI Hydrocel channel names
            _fix_EGI_channels(raw, montage_name)

        # Set Montage
        if montage_file is not None:
            montage = mne.channels.read_custom_montage(montage_file)
            raw.set_montage(montage)
        else:
            raw.set_montage(montage_name)

    return raw, montage


def process_EEG_data(raw, EEG_PARAMETERS):
    """
    MNE Raw Object Preprocessing

    EEG_PARAMETERS:
        'Montage':  MNE compatible string name of Montage (e.g. 'GSN-HydroCel-129')
        'Channels': '10-20', 'All'
        'Crop Time': [tmin, tmax] (sec)
        'Filtering': [Low-cut freq, High-cut freq] (Hz)
        'Notch Freq': Notch Filtering frequency, including harmonics (Hz)
        'Sfreq': (Hz)
        'Epoch Size': (sec)
        'Epoch Overlap': Overlap percentage (0% - 99%)
        'Epoch Baseline Correction': (boolean)
        'Epoch Peak-to-Peak Threshold': Peak-to-peak amplitude Threshold (V) or None
        'Interpolate Bad Channels': Bad if signal is flat, or If 20% of the signal exceeds p-t-p threshold (boolean)
        'Epoch Rejection': Reject epoch if Peak-to-peak threshold exceeds in more than 20% of the channels
        'Reference': 'Default', 'Average', 'Cz', 'Frontal' (Fp1-F3-Fz-F4-Fp2)

    Args:
        raw, (MNE structure): the raw data
        EEG_PARAMETERS, (dict): Dictionary with EEG Parameters

    Returns:
        raw, (MNE structure) - the preprocessed raw structure
        epochs, (MNE structure) - the epoched structure
    """

    # Pick EEG channels
    if ('Channels' in EEG_PARAMETERS.keys()):
        if (EEG_PARAMETERS['Channels'] is not None):

            if (EEG_PARAMETERS['Channels'] == 'All'):
                raw.pick_types(eeg=True, meg=False, stim=False, eog=False, ecg=False, emg=False)

            elif (EEG_PARAMETERS['Channels'] == 'All_excl'):
                if ('Montage' in EEG_PARAMETERS.keys()):
                    if (EEG_PARAMETERS['Montage'] in ['GSN-HydroCel-256', 'GSN-HydroCel-257']):
                        raw.pick_types(eeg=True, meg=False, stim=False, eog=False, ecg=False, emg=False)
                        raw.drop_channels(EGI257_excl)
                    else:
                        raw.pick_types(eeg=True, meg=False, stim=False, eog=False, ecg=False, emg=False)
                else:
                    raw.pick_types(eeg=True, meg=False, stim=False, eog=False, ecg=False, emg=False)

            elif (EEG_PARAMETERS['Channels'] == '10-20'):  # Renamed and Reordered!
                if ('Montage' in EEG_PARAMETERS.keys()):
                    if (EEG_PARAMETERS['Montage'] in ['GSN-HydroCel-256', 'GSN-HydroCel-257']):
                        raw.pick_types(eeg=True, meg=False, stim=False, eog=False, ecg=False, emg=False)
                        raw.pick_channels(EGI257_10_20)
                        raw.rename_channels(mapping=EGI257_10_20_map)
                        raw.reorder_channels(EEG_10_20)
                    elif (EEG_PARAMETERS['Montage'] in ['GSN-HydroCel-128', 'GSN-HydroCel-129']):
                        raw.pick_types(eeg=True, meg=False, stim=False, eog=False, ecg=False, emg=False)
                        raw.pick_channels(EGI129_10_20)
                        raw.rename_channels(mapping=EGI129_10_20_map)
                        raw.reorder_channels(EEG_10_20)
                    else:
                        raw.pick_types(eeg=True, meg=False, stim=False, eog=False, ecg=False, emg=False)
                        # Check if EEG_10_20 names exist in Raw, and select (20)
                        if all([ch in raw.ch_names for ch in EEG_10_20]):
                            raw.pick_channels(EEG_10_20)
                            raw.reorder_channels(EEG_10_20)
                        # Else, Check if EXIMIA_10_20 names exist in Raw, and select (20)
                        elif all([ch in raw.ch_names for ch in EXIMIA_10_20]):
                            raw.pick_channels(EXIMIA_10_20)
                            raw.reorder_channels(EXIMIA_10_20)
                        # Else, Pick the ones that appear in EEG_10_20
                        else:
                            raw.pick_channels(EEG_10_20)
                            raw.reorder_channels(EEG_10_20)
                else:
                    raw.pick_types(eeg=True, meg=False, stim=False, eog=False, ecg=False, emg=False)
                    raw.pick_channels(EEG_10_20)
                    raw.reorder_channels(EEG_10_20)

    # Crop EEG from tmin to tmax (sec)
    if ('Crop Time' in EEG_PARAMETERS.keys()):
        if (EEG_PARAMETERS['Crop Time'] is not None):
            raw.crop(EEG_PARAMETERS['Crop Time'][0], EEG_PARAMETERS['Crop Time'][1])

    # EEG Filtering (firwin design)
    if ('Filtering' in EEG_PARAMETERS.keys()):
        if (EEG_PARAMETERS['Filtering'] is not None):
            raw.filter(EEG_PARAMETERS['Filtering'][0], EEG_PARAMETERS['Filtering'][1], fir_design='firwin')

    if ('Notch Freq' in EEG_PARAMETERS.keys()):
        if (EEG_PARAMETERS['Notch Freq'] is not None):
            frequencies = np.arange(EEG_PARAMETERS['Notch Freq'], (EEG_PARAMETERS['Sfreq'] / 2) + 1,
                                    EEG_PARAMETERS['Notch Freq'])
            if frequencies != []:
                raw.notch_filter(frequencies, fir_design='firwin')

    # Resampling
    if ('Sfreq' in EEG_PARAMETERS.keys()):
        if (EEG_PARAMETERS['Sfreq'] is not None):
            raw.resample(sfreq=EEG_PARAMETERS['Sfreq'], npad='auto')

    epochs = None

    # EEG Epoching
    if ('Epoch Size' in EEG_PARAMETERS.keys()):
        if (EEG_PARAMETERS['Epoch Size'] is not None):

            epoch_samples = EEG_PARAMETERS['Epoch Size'] * raw.info['sfreq']

            if ('Epoch Overlap' in EEG_PARAMETERS.keys()):
                if (EEG_PARAMETERS['Epoch Overlap'] is not None):  # % percentage overlap

                    p = EEG_PARAMETERS['Epoch Overlap']  # percentage (0 - 99%)
                    c = (100 - p) / 100  # calculate coefficient and round to 3 decimal places
                    # No of epochs to remove from the end (no full epoch_sample window)
                    e_r = (epoch_samples / (epoch_samples * c)) - 1

                    events = np.array([[int((epoch_samples * c * i)) + raw.first_samp, 0, 1] for i in
                                       range(0, int((len(raw) / (epoch_samples * c)) - e_r))])
                else:
                    events = np.array([[int((epoch_samples * i)) + raw.first_samp, 0, 1] for i in
                                       range(0, int((len(raw)) / epoch_samples))])

            else:
                events = np.array([[int((epoch_samples * i)) + raw.first_samp, 0, 1] for i in
                                   range(0, int((len(raw)) / epoch_samples))])

            event_id = dict(start=1)

            if ('Epoch Baseline Correction' in EEG_PARAMETERS.keys()):
                if (EEG_PARAMETERS['Epoch Baseline Correction'] == True):
                    epochs = mne.Epochs(raw, events, event_id, tmin=0, tmax=EEG_PARAMETERS['Epoch Size'],
                                        baseline=(None, 0), reject=None, flat=None, reject_by_annotation=False,
                                        preload=True)
                else:
                    epochs = mne.Epochs(raw, events, event_id, tmin=0, tmax=EEG_PARAMETERS['Epoch Size'],
                                        baseline=None, reject=None, flat=None, reject_by_annotation=False, preload=True)
            else:
                epochs = mne.Epochs(raw, events, event_id, tmin=0, tmax=EEG_PARAMETERS['Epoch Size'],
                                    baseline=None, reject=None, flat=None, reject_by_annotation=False, preload=True)

            # Make sure number of samples are round at 'Epoch size' * 'Sfreq'
            epochs._data = epochs._data[:, :, :EEG_PARAMETERS['Epoch Size'] * EEG_PARAMETERS['Sfreq']]

    reject = None
    if ('Epoch Peak-to-Peak Threshold' in EEG_PARAMETERS.keys()):
        if (EEG_PARAMETERS['Epoch Peak-to-Peak Threshold'] is not None):
            reject = dict(eeg=EEG_PARAMETERS['Epoch Peak-to-Peak Threshold'])

    if ('Interpolate Bad Channels' in EEG_PARAMETERS.keys()):
        if (EEG_PARAMETERS['Interpolate Bad Channels'] == True):

            print('Interpolating Bad Channels...')
            bad_channels = _find_bad_channels(epochs.copy(), reject,
                                             threshold=0.2)  # Bad, if 20% of timeseries is rejected or flat
            if (bad_channels != []):
                print('Bad channels have been identified and interpolated: ' + str(bad_channels))
                raw.info['bads'] = bad_channels
                epochs.info['bads'] = bad_channels
                raw.interpolate_bads()
                epochs.interpolate_bads()
            else:
                print('No Bad Channels!')

    if ('Epoch Rejection' in EEG_PARAMETERS.keys()):
        if (EEG_PARAMETERS['Epoch Rejection'] == True):
            if reject is not None:  # If Peak-to-peak threshold is set

                print('Epoch rejection by peak-to-peak thresholds...')
                bad_epochs = _find_bad_epochs(epochs.copy(), reject, threshold=0.2)
                if (bad_epochs != []):
                    epochs.drop(indices=bad_epochs)
                    print('Removed epoch indices: ' + str(bad_epochs))
                else:
                    print('No Bad Epochs!')

    # Set Reference
    if ('Reference' in EEG_PARAMETERS.keys()):
        if (EEG_PARAMETERS['Reference'] is not None):

            if (EEG_PARAMETERS['Reference'] == 'Default'):
                epochs.set_eeg_reference(ref_channels=[])
                raw.set_eeg_reference(ref_channels=[], verbose=False)

            elif (EEG_PARAMETERS['Reference'] == 'Average'):
                epochs.set_eeg_reference(ref_channels='average',
                                         projection=False)  # proj=False to directly apply the projection
                raw.set_eeg_reference(ref_channels='average',
                                      projection=False, verbose=False)  # proj=False to directly apply the projection

            elif (EEG_PARAMETERS['Reference'] == 'Cz'):
                channel_name = 'Cz'
                if (EEG_PARAMETERS['Montage'] in ['GSN-HydroCel-256', 'GSN-HydroCel-257'] and EEG_PARAMETERS[
                    'Channels'] != '10-20'):
                    channel_name = EGI257_10_20_inv_map[channel_name]
                elif (EEG_PARAMETERS['Montage'] in ['GSN-HydroCel-128', 'GSN-HydroCel-129'] and EEG_PARAMETERS[
                    'Channels'] != '10-20'):
                    channel_name = EGI129_10_20_inv_map[channel_name]

                epochs.set_eeg_reference(ref_channels=[channel_name],
                                         projection=False)  # proj=False to directly apply the projection
                raw.set_eeg_reference(ref_channels=[channel_name],
                                      projection=False, verbose=False)  # proj=False to directly apply the projection

                # Interpolate the new reference
                raw.info['bads'] = [channel_name]
                epochs.info['bads'] = [channel_name]
                raw.interpolate_bads()
                epochs.interpolate_bads()

            elif (EEG_PARAMETERS['Reference'] == 'Frontal'):
                channel_names = ['Fp1', 'F3', 'Fz', 'F4', 'Fp2']
                if (EEG_PARAMETERS['Montage'] in ['GSN-HydroCel-256', 'GSN-HydroCel-257'] and EEG_PARAMETERS[
                    'Channels'] != '10-20'):
                    channel_names = [EGI257_10_20_inv_map[c] for c in channel_names]
                elif (EEG_PARAMETERS['Montage'] in ['GSN-HydroCel-128', 'GSN-HydroCel-129'] and EEG_PARAMETERS[
                    'Channels'] != '10-20'):
                    channel_names = [EGI129_10_20_inv_map[c] for c in channel_names]

                epochs.set_eeg_reference(ref_channels=channel_names,
                                         projection=False)  # proj=False to directly apply the projection
                raw.set_eeg_reference(ref_channels=channel_names,
                                      projection=False, verbose=False)  # proj=False to directly apply the projection

    return raw, epochs



def exponential_moving_standardization(epoched_data, sfreq, init_block_size=4):
    """
        Performs exponential moving standardization (EMS) on Epoched EEG data

        EMS updates the statistics (mean/var) of standardization recursively throughout time by using a decay factor
        (alpha). Standardization is performed independently on every channel (channel-wise).

        A detailed description of the method can be found in (Schirrmeister et al, 2017)

        Args:
            epoched_data, (ndarray): the epoched eeg data (epochs, channels, samples)
            sfreq (int): The sampling frequency (hz)
            init_block_size (int): initial window of for estimating statistics (seconds)
        Returns:
            epoched_data_EMS, (ndarray): the standardized epoched eeg data
        """

    print('Performing Channel-wise Exponential moving Standardization...')

    alpha = 0.25 / sfreq  # Alpha value based on (Schirrmeister et al, 2017)

    # Initial block size to estimate mean/var
    if (init_block_size is not None):
        init_block_size = init_block_size * sfreq  # sec to samples

    # Concatenate epochs
    no_epochs = epoched_data.shape[0]
    eeg_data = np.hstack(tuple(epoched_data))
    # Transpose
    eeg_data = eeg_data.T

    # Compute Exponential moving Standardization
    epoched_data_EMS = _exponential_running_standardize(eeg_data, factor_new=alpha,
                                                      init_block_size=init_block_size, eps=1e-4)

    # Transpose
    epoched_data_EMS = epoched_data_EMS.T
    # Split epochs
    epoched_data_EMS = np.array(np.hsplit(epoched_data_EMS, no_epochs))

    return epoched_data_EMS


#-----------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------- INTERNAL ----------------------------------------------------------

def _fix_EGI_channels(raw, montage_name):
    """
    Fixes EGI channel naming from GSN HydroCel Montages (128/129, 256/267 Channels), when Cz is missing and channel
    names have been changed to 10-20 names
    """

    channel_names = raw.info['ch_names']

    # Ads Cz channel if it doesn't exist
    if not ((montage_name in ['GSN-HydroCel-128', 'GSN-HydroCel-129']
             and ('E129' in channel_names or 'Cz' in channel_names))
            or (montage_name in ['GSN-HydroCel-256', 'GSN-HydroCel-257']
                and ('E257' in channel_names or 'Cz' in channel_names))):
        info = mne.create_info(['Cz'], raw.info['sfreq'], ch_types='eeg')
        data = np.zeros((1, raw.n_times))
        Cz = mne.io.RawArray(data, info)
        raw.add_channels([Cz])

    # Check for Channel name consistency
    channel_names = raw.info['ch_names']
    fixed_channel_names = channel_names

    if (montage_name in ['GSN-HydroCel-128', 'GSN-HydroCel-129']):
        fixed_channel_names = [EGI129_10_20_inv_map[name] if name in EGI129_10_20_inv_map.keys() else name
                               for name in channel_names]
    elif (montage_name in ['GSN-HydroCel-256', 'GSN-HydroCel-257']):
        fixed_channel_names = [EGI257_10_20_inv_map[name] if name in EGI257_10_20_inv_map.keys() else name
                               for name in channel_names]

    # Rename, if channel names contain 10-20 names
    if (channel_names != fixed_channel_names):
        dict_mapping = dict(zip(channel_names, fixed_channel_names))
        raw.rename_channels(dict_mapping)

        # Sort fixed channel names
        sorted_fixed_channel_names = natsorted(fixed_channel_names)

        # Reorder, if channel names are not in order
        if (fixed_channel_names != sorted_fixed_channel_names):
            raw.reorder_channels(sorted_fixed_channel_names)

    return None


def _find_bad_channels(epochs, reject, threshold=0.2):
    """
    Finds bad channels from an epoch structure, based on min/max peak-to-peak amplitudes of epochs (flatness/rejection)
    If percentage of epochs exceeds threshold, channel is marked as 'bad'
    """

    bad_channels = []

    no_epochs = len(epochs)
    flat = dict(eeg=1e-12)  # Flatness is defined with maximum acceptable peak-to-peak of 1e-12

    epochs.drop_bad(reject=reject, flat=flat, verbose=False)
    epochs_drop_log = [ch_list for ch_list in epochs.drop_log if (ch_list != ['NO_DATA'] and ch_list != [])]

    channel_counter = dict(zip(epochs.ch_names, np.zeros(len(epochs.ch_names))))

    for epoch in epochs_drop_log:
        for channel in epoch:
            if channel in epochs.ch_names:
                channel_counter[channel] += 1

    for channel in channel_counter.keys():
        if (channel_counter[channel] / no_epochs > threshold):
            bad_channels.append(channel)

    return bad_channels


def _find_bad_epochs(epochs, reject, threshold=0.2):
    """
    Finds bad epochs from an epoch structure, based on peak-to-peak thresholds (rejection)
    If percentage of channels exceed threshold, epoch is marked as 'bad'
    """

    bad_epochs = []
    no_channels = len(epochs.ch_names)

    epochs.drop_bad(reject=reject, flat=None, verbose=False)
    epochs_drop_log = [ch_list for ch_list in epochs.drop_log if ch_list != ['NO_DATA']]

    for i, epoch in enumerate(epochs_drop_log):
        if len(epoch) / no_channels > threshold:
            bad_epochs.append(i)

    return bad_epochs


def _exponential_running_standardize(data, factor_new=0.001,
                                     init_block_size=None, eps=1e-4):
    """
    BrainDecode Method (https://braindecode.org/)
    """

    adjust = True  # See pf.ewm for details

    df = pd.DataFrame(data)
    meaned = df.ewm(alpha=factor_new, adjust=adjust).mean()
    demeaned = df - meaned
    squared = demeaned * demeaned
    square_ewmed = squared.ewm(alpha=factor_new, adjust=adjust).mean()
    standardized = demeaned / np.maximum(eps, np.sqrt(np.array(square_ewmed)))
    standardized = np.array(standardized)
    if init_block_size is not None:
        # other_axis = tuple(range(1, len(data.shape)))
        init_mean = np.mean(data[0:init_block_size], axis=0,
                            keepdims=True)
        init_std = np.std(data[0:init_block_size], axis=0,
                          keepdims=True)
        init_block_standardized = (data[0:init_block_size] - init_mean) / \
                                  np.maximum(eps, init_std)
        standardized[0:init_block_size] = init_block_standardized
    return standardized
