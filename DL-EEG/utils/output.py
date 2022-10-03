# Author: Konstantinos Patlatzoglou <konspatl@gmail.com>

import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from utils import utils


# ------------------------------------------------- OUTPUT UTIL --------------------------------------------------------
linewidth = 2  # Matplotlib parameter

#------------------------------------------------ PLOT FUNCTIONS -------------------------------------------------------

def plot_predictions(subject, predictions, states, targets, ylim=None, target_values=None, CLASSES=None,
                     states_colors=None, class_colors=None, targets_cmap=None):
    """
    Plots subject predictions

    Args:
        subject (str): name of subject
        predictions (ndarray): the predictions Array (state,)(epoch, target)
        states (list): list of state names
        targets (str or list): '1-hot' (classification) or list of target names (regression)
        ylim (tuple or None): y plot limits (min, max)
        target_values (list or None): list of target values per state
        CLASSES (list or None): list of class names
        state_colors (list or None): list of state colors (Matplotlib arg)
        class_colors (list or None): list of class colors (Matplotlib arg)
        targets_cmap (list or None): list of target colormaps (Matplotlib arg)
    Returns:
        figure (matplotlib.Figure): a matplotlib figure
    """

    # State Info
    state_epochs = [state.shape[0] for state in predictions]
    epochs = np.arange(1, np.sum(state_epochs) + 1)

    # Predictions and Targets
    Ypred = np.vstack(tuple(predictions))

    Yte = None
    if target_values is not None:  # If Target Values
        state_targets = utils.create_state_targets(state_epochs, target_values)
        Yte = np.vstack(tuple(state_targets))

    if ylim is None:
        y_min = np.min(Ypred)
        y_max = np.max(Ypred)
        ylim = (int(y_min), int(y_max))

    # Color Info
    if states_colors is None:
        states_colors = [None for state in states]
    if CLASSES is not None and class_colors is None:
        class_colors = [None for _class in CLASSES]

    # -----------------------------------Plotting--------------------------------
    figure = plt.figure(figsize=(15, 5))

    # Plot Vertical Dashed Lines for each State
    s_start = 0
    for s, state in enumerate(states):
        s_end = np.sum(state_epochs[0:(s + 1)])
        plt.axvline(x=s_end, linestyle='--', color='black', label='_nolegend_')
        text_height = ylim[1] * 0.85 if (s % 2 == 0) else ylim[1] * 0.75
        plt.text(x=s_start, y=text_height, s=state, color=states_colors[s],
                 path_effects=[pe.Stroke(linewidth=1, foreground='black'), pe.Normal()])
        s_start = s_end

    # Plot Predictions
    if targets == '1-hot': # Classification

        for c, class_name in enumerate(CLASSES):  # For each Class
            plt.plot(epochs, Ypred[:, c], color=class_colors[c], alpha=0.5, linewidth=linewidth)
            if Yte is not None:
                plt.plot(epochs, Yte[:, c], '--', color=class_colors[c], alpha=0.9, linewidth=linewidth,
                         label='_nolegend_')

    else:  # Regression
        for t, target in enumerate(targets):  # For each Target
            if targets_cmap is None:
                plt.plot(epochs, Ypred[:, t], color='grey', alpha=0.6, linewidth=linewidth)
            else:
                plt.scatter(epochs, Ypred[:, t], c=Ypred[:, t], marker='.', cmap=targets_cmap[t],
                            vmin=ylim[0], vmax=ylim[1], alpha=0.6, linewidth=linewidth)
            if Yte is not None:
                s_start = 0
                for s, state in enumerate(states):
                    s_end = np.sum(state_epochs[0:(s + 1)])
                    plt.plot(epochs[s_start:s_end], Yte[s_start:s_end, t], '--', color=states_colors[s], alpha=0.9,
                             linewidth=linewidth, label='_nolegend_')
                    s_start = s_end

    # Add Labels
    plt.xlabel('EEG Epoch')
    plt.ylabel(str(targets))
    plt.ylim(ylim)
    if targets == '1-hot':
        plt.legend(CLASSES, fontsize=12)
    else:
        plt.legend(targets, fontsize=12)
    plt.title(subject + ' Predictions')
    plt.tight_layout()

    return figure


def plot_history(subject, history, loss=None, metrics=None):
    """
    Plots subject history.

    Args:
        subject (str): name of subject
        history (dict): history structure (tensorflow/keras history.history)
        loss (str): name of Loss Function
        metrics (list): list of metrics
    Returns:
        figure (matplotlib.Figure): a matplotlib figure
    """

    # History Info
    training_epochs = np.arange(1, len(history['val_loss']) + 1)
    loss_keys = [key for key in history.keys() if 'loss' in key]
    metrics_keys = [key for key in history.keys() if 'loss' not in key]

    # -----------------------------------Plotting--------------------------------
    figure = plt.figure(figsize=(15, 5))

    # Loss Plot
    plt.subplot(121)
    legend_labels = []
    for loss_key in loss_keys:
        if 'val' in loss_key: # Validation Loss
            plt.plot(training_epochs, history[loss_key], color='cyan', linewidth=linewidth)
        else: # Training Loss
            plt.plot(training_epochs, history[loss_key], color='violet', linestyle='--', linewidth=linewidth)
        legend_labels.append(loss_key)


    plt.xlabel('Training Epoch')
    if loss is None:
        plt.ylabel('Loss')
    else:
        plt.ylabel(loss)
    plt.legend(legend_labels, loc='upper right')
    plt.title(subject + ' Loss')

    # Metrics Plot
    plt.subplot(122)
    legend_labels = []
    for metrics_key in metrics_keys:
        if 'val' in metrics_key:  # Validation Loss
            plt.plot(training_epochs, history[metrics_key], color='dodgerblue', linewidth=linewidth)
        else:  # Training Loss
            plt.plot(training_epochs, history[metrics_key], color='purple', linestyle='--', linewidth=linewidth)
        legend_labels.append(metrics_key)

    plt.xlabel('Training Epoch')
    if metrics is None:
        plt.ylabel('Metric')
    else:
        plt.ylabel(str(metrics))
    plt.legend(legend_labels, loc='upper right')
    plt.title(subject + ' Metrics')
    plt.tight_layout()

    return figure
    # ---------------------------------------------------------------------------


def plot_confusionMatrix(subject, confusionMatrix, CLASSES, normalize=False):
    """
    Plots subject confusion Matrix.

    Args:
        subject (str): name of subject
        confusionMatrix (ndarray): the confusion matrix array
        CLASSES (list) list of class names
        normalize (boolean): if True, confusion matrix is normalized across instances
    Returns:
        figure (matplotlib.Figure): a matplotlib figure
    """
    if normalize:
        confusionMatrix = confusionMatrix.astype('float') / confusionMatrix.sum(axis=1)[:, np.newaxis]

    # -----------------------------------Plotting--------------------------------
    figure = plt.figure(figsize=(8, 8))

    plt.imshow(confusionMatrix, interpolation='nearest', cmap=plt.cm.Blues)

    fmt = '.2f' if normalize else 'd'
    thresh = confusionMatrix.max() / 2.
    for i, j in itertools.product(range(confusionMatrix.shape[0]), range(confusionMatrix.shape[1])):
        plt.text(j, i, format(confusionMatrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if confusionMatrix[i, j] > thresh else "black")

    tick_marks = np.arange(len(CLASSES))
    plt.xticks(tick_marks, CLASSES, rotation=45)
    plt.yticks(tick_marks, CLASSES)
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title(subject + ' Confusion matrix')
    plt.tight_layout()

    return figure
    # ---------------------------------------------------------------------------


def plot_all_scores(subjects, scores_list, loss, metrics):
    """
    Plots all subject scores.

    Args:
        subjects (list): list of subject names
        scores_list (list): list of subject scores
        loss (str): name of loss function
        metrics (list): list of metric names
    Returns:
        figure (matplotlib.Figure): a matplotlib figure
    """

    # Score Info
    score_loss_list = [score['loss'] for score in scores_list]

    metric = metrics[0]
    if metric in scores_list[0]:
        score_metric_list = [score[metric] for score in scores_list]
    else:
        score_metric_list = []
        metric_keys = [key for key in scores_list[0] if metric in key]
        for metric_key in metric_keys:
            score_metric_list.append([score[metric_key] for score in scores_list])
        score_metric_list = list(np.mean(score_metric_list, axis=0))

    # -----------------------------------Plotting--------------------------------
    figure = plt.figure(figsize=(15, 5))

    # Loss Plot
    plt.subplot(121)
    plt.bar(subjects, score_loss_list, align='center', color='cyan')
    plt.bar(len(subjects), np.mean(score_loss_list), yerr=np.std(score_loss_list),
            align='center', color='darkcyan')
    plt.text(len(subjects), np.mean(score_loss_list), '{:.2f}'.format(np.mean(score_loss_list)))

    plt.xlabel('Subjects')
    plt.ylabel(loss)
    plt.xticks(np.arange(len(subjects) + 1), np.append(subjects, 'Average'), rotation=30, fontsize=12)
    plt.title(loss + ' loss')

    # Metric Plot
    plt.subplot(122)
    plt.bar(subjects, score_metric_list, align='center', color='dodgerblue')
    plt.bar(len(subjects), np.mean(score_metric_list), yerr=np.std(score_metric_list),
            align='center', color='blue')
    plt.text(len(subjects), np.mean(score_metric_list), '{:.2f}'.format(np.mean(score_metric_list)))

    plt.xlabel('Subjects')
    plt.ylabel(metric)
    plt.xticks(np.arange(len(subjects) + 1), np.append(subjects, 'Average'), rotation=30, fontsize=12)
    plt.title(metric)
    plt.tight_layout()

    return figure
    # ---------------------------------------------------------------------------