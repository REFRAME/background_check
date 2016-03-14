import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from prg import calculate_prg_points

"""
This file contains the function that calculates the volume under the recall-gain_1 and roc_2 * precision-gain1 surface.

"""


def calculate_volume(step1_labels, step1_scores, step2_labels, step2_scores):
    """This function calculates the volume under the recall-gain_1 and roc_2 * precision-gain1 surface.

    Args:
        step1_labels ([int]): The labels for evaluating the training data vs reject data classifier.
            1 for training data, 0 for reject data.
        step1_scores ([float]): The second parameter. Defaults to None.
            Second line of description should be indented.
        step2_labels ([int]): The labels for evaluating the real training data classifier.
            1 for the positive class, 0 for the negative class.
        step2_scores ([float]): The scores obtained from the second classifier.

    Returns:
        float: The volume under the recall-gain_1 and roc_2 * precision-gain1 surface.

    .. _PEP 484:
       https://www.python.org/dev/peps/pep-0484/

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

    print areas
    return auc(areas, recall_gains, reorder=True)

