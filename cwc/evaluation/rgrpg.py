import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from prg import calculate_prg_points
import matplotlib.pyplot as plt

"""
This file contains functions for calculating the rgrpg surface, plotting it and calculating the volume under it.

"""


def build_rgrpg_surface(step1_labels, step1_scores, step2_labels, step2_scores):
    """This function finds the points that form the rgrpg surface.

    Args:
        step1_labels ([int]): The labels for evaluating the training data vs reject data classifier.
            1 for training data, 0 for reject data.
        step1_scores ([float]): The second parameter. Defaults to None.
            Second line of description should be indented.
        step2_labels ([int]): The labels for evaluating the real training data classifier.
            1 for the positive class, 0 for the negative class.
        step2_scores ([float]): The scores obtained from the second classifier.

    Returns:
        [float]: The recall-gains calculated by thresholding over step1_scores and using step1_labels.
        [float]: The areas of the roc curves calculated by thresholding over step2_scores and using step2_labels,
            multiplied by the precision-gains corresponding to the returned recall-gains.

    """

    prg_curve = calculate_prg_points(step1_labels, step1_scores)
    recall_gains = prg_curve['recall_gain'][prg_curve['recall_gain'] >= 0]
    precision_gains = prg_curve['precision_gain'][prg_curve['recall_gain'] >= 0]
    pos_scores = prg_curve['pos_score'][prg_curve['recall_gain'] >= 0]

    n_recalls = np.alen(recall_gains)
    areas = np.zeros(n_recalls)

    for rg in np.arange(n_recalls):
        true_positive_indices = np.logical_and(step1_scores >= pos_scores[rg], step1_labels == 1)
        probabilities = step2_scores[true_positive_indices]
        labels = step2_labels[true_positive_indices] == 1

        areas[rg] = roc_auc_score(labels, probabilities)*precision_gains[rg]

    return [recall_gains, areas]


def plot_rgrpg_2d(recall_gains, areas):
    """This function plots the 2d version of the rgrpg surface, with the recall-gains from the
    training data vs reject data classifier on the x-axis and the area under the corresponding roc curve of the real
    training data classifier, multiplied by the corresponding precision-gain from the
    training data vs reject data classifier on the y-axis.

    Args:
        recall_gains ([float]): The recall-gains from the rgrpg surface.
        areas ([float]): The areas of the corresponding roc curves, multiplied by the corresponding precision-gains.

    Returns:
        Nothing.

    """

    plt.scatter(recall_gains, areas)
    plt.plot(recall_gains, areas)
    plt.xlabel("Recall-gains_1")
    plt.ylabel("AUROC_2 * Precision-gains_1")
    axes = plt.gca()
    axes.set_xlim([0.0, 1.0])
    axes.set_ylim([0.0, 1.0])
    plt.show()


def calculate_volume(recall_gains, areas):
    """This function calculates the volume under the rgrpg surface.

    Args:
        recall_gains ([float]): The recall-gains from the rgrpg surface.
        areas ([float]): The areas of the corresponding roc curves, multiplied by the corresponding precision-gains.

    Returns:
        float: The volume under the recall-gain_1 and roc_2 * precision-gain1 surface.

    """
    return auc(areas, recall_gains, reorder=True)

