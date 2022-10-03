# Author: Konstantinos Patlatzoglou <konspatl@gmail.com>

import os
from pathlib import Path

import numpy as np
import json
import matplotlib

from utils import utils
from utils import output

# ------------------------------EXPORT FIGURES PARAMETERS-------------------------------
RESULTS_DIR_NAME = 'Model Testing'


STATE_COLORS = ['red', 'blue']  # Matplotlib color names
CLASSES_COLORS = ['red', 'blue'] # Matplotlib color names

matplotlib.rcParams.update({'font.size': 14})
# --------------------------------------------------------------------------------------

def main():

    current_path = Path.cwd()

    # ----------------------- IMPORT EXPERIMENT PARAMETERS --------------------------
    print('Importing Experiment Parameters...')

    result_path = current_path.parent / 'results' / RESULTS_DIR_NAME
    if not os.path.exists(result_path):
        print('Results Directory is missing')
        exit(1)

    EEG_DATASET_PARAMETERS = []
    for eeg_dataset_parameters_file in list(result_path.glob('EEG_DATASET_PARAMETERS_*.json')):
        EEG_DATASET_PARAMETERS.append(json.load(open(str(eeg_dataset_parameters_file))))

    MODEL_PARAMETERS = json.load(open(str(result_path / 'MODEL_PARAMETERS.json')))
    MODEL_INFO = json.load(open(str(result_path / 'MODEL_INFO.json')))

    CLASSES = None
    if MODEL_INFO['Targets'] == '1-hot':
        CLASSES = json.load(open(str(result_path / 'CLASSES.json')))

    # -------------------------------------------------------------------------------

    # ---------------------------- EEG DATASET ANALYSIS -----------------------------
    print('Exporting Dataset Figures...')

    dataset_export_paths = sorted([path for path in result_path.iterdir() if path.is_dir()])

    # For Each EEG Dataset Directory
    for d, dataset_export_path in enumerate(dataset_export_paths):
        subject_info_file_list = sorted(list(dataset_export_path.glob('*_info.json')))

        subjects = []
        predictions_list = []
        subject_info_list = []

        # For Each Subject
        for subject_info_file in subject_info_file_list:

            subject = subject_info_file.name.replace('_info.json', '')
            predictions = np.load(str(dataset_export_path / (subject + '_predictions.npy')), allow_pickle=True)
            subject_info = json.load(open(str(subject_info_file)))

            # Export Predictions Figure
            fig = output.plot_predictions(subject, predictions, subject_info['States'], MODEL_INFO['Targets'],
                                          target_values=subject_info['Target Values'], CLASSES=CLASSES,
                                          states_colors=STATE_COLORS, class_colors=CLASSES_COLORS)
            fig.savefig(str(dataset_export_path / (subject + '_predictions.png')))
            fig.clf()

            # Export History Figure
            if subject_info['History'] is not None:
                fig = output.plot_history(subject, subject_info['History'], loss=MODEL_PARAMETERS['Loss'],
                                          metrics=MODEL_PARAMETERS['Metrics'])
                fig.savefig(str(dataset_export_path / (subject + '_history.png')))
                fig.clf()

            # Export Confusion Matrix (Classification / Target Values)
            if MODEL_INFO['Targets'] == '1-hot' and subject_info['Target Values'] is not None:
                confusionMatrix = utils.confusionMatrix(predictions, subject_info['Target Values'], len(CLASSES))

                fig = output.plot_confusionMatrix(subject, confusionMatrix, CLASSES, normalize=False)
                fig.savefig(str(dataset_export_path / (subject + '_CM.png')))
                fig.clf()

            subjects.append(subject)
            predictions_list.append(predictions)
            subject_info_list.append(subject_info)


        if EEG_DATASET_PARAMETERS[d]['EEG_DATASET']['Target Values'] is not None:
            # Get Scores List
            scores_list = [subject_info['Score'] for subject_info in subject_info_list]

            fig = output.plot_all_scores(subjects, scores_list, MODEL_PARAMETERS['Loss'], MODEL_PARAMETERS['Metrics'])
            fig.savefig(str(dataset_export_path / 'All Scores.png'))
            fig.clf()
    # -------------------------------------------------------------------------------


if __name__ == '__main__':
    main()