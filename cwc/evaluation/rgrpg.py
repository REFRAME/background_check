from __future__ import division
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from prg import calculate_prg_points
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings


"""
This file contains the class RGRPG, that represents the Recall-gain-ROC-Precision-gain surface,  with methods for
plotting the surface in 2D or 3D and for calculating the volume under it.

"""


class RGRPG:
    """This class represents a Recall-gain-ROC-Precision-gain (RGRPG) surface. An object of class RGRPG is built based
    on the result of two models:

    1- The first one is a training data vs reject data classifier and its recall-gain and precision-gain values at
        various thresholds are used to build the surface;
    2- The second (binary) classifier was trained to separate both classes from the original training data and its
        ROC curves for all recall-gain values of the first classifier are used to build the surface.

    Args:
        step1_reject_scores ([float]): Scores for the reject data, obtained from
            the training data vs reject data classifier.
        step1_training_scores ([float]): Scores for the training data, obtained from
            the training data vs reject data classifier.
        step2_training_scores ([int]): Scores for the training data, obtained from
            the original classifier.
        training_labels ([int]): Labels of the training data. 1 for the positive class
            and 0 for the negative class.

    Attributes:
        recall_gains ([float]): The recall-gains calculated by thresholding over step1_scores and using step1_labels.
        precision_gains ([float]): The precision-gains calculated by thresholding over step1_scores
            and using step1_labels.
        rocs (dict{[[float]]}): The ROC curves calculated by thresholding over step2_scores and using step2_labels.
        areas ([float]): The areas of the roc curves calculated by thresholding over step2_scores and using
            step2_labels.

    """
    def __init__(self, step1_reject_scores, step1_training_scores, step2_training_scores, training_labels):
        step1_scores = np.append(step1_training_scores, step1_reject_scores)
        step1_labels = np.append(np.ones(np.alen(training_labels)), np.zeros(np.alen(step1_reject_scores)))

        prg_curve = calculate_prg_points(step1_labels, step1_scores)
        self.recall_gains = prg_curve['recall_gain'][prg_curve['recall_gain'] >= 0]
        self.precision_gains = prg_curve['precision_gain'][prg_curve['recall_gain'] >= 0]
        pos_scores = prg_curve['pos_score'][prg_curve['recall_gain'] >= 0]

        n_recalls = np.alen(self.recall_gains)
        self.areas = np.zeros(n_recalls)

        self.rocs = dict()

        for rg in np.arange(n_recalls):
            true_positive_indices = np.where(np.logical_and(step1_scores >= pos_scores[rg], step1_labels == 1))[0]
            probabilities = step2_training_scores[true_positive_indices]
            labels = training_labels[true_positive_indices]

            self.areas[rg] = roc_auc_score(labels, probabilities)
            fpr, tpr, thresholds = roc_curve(labels, probabilities)
            self.rocs[rg] = np.append(tpr.reshape(-1, 1), fpr.reshape(-1, 1), axis=1)
        self.rocs = even_out_roc_points(self.rocs)

    def plot_rgrpg_2d(self, fig=None):
        """This method plots the 2d version of the RGPRG surface, with the recall-gains from the
        training data vs reject data classifier on the x-axis and the area under the corresponding roc curve of the real
        training data classifier, multiplied by the corresponding precision-gain from the
        training data vs reject data classifier on the y-axis.

        Args:
            None.

        Returns:
            Nothing.

        """
        # Ignore warnings from matplotlib
        warnings.filterwarnings("ignore")
        if fig == None:
            fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.recall_gains, self.areas*self.precision_gains, 'bo-')
        ax.set_xlabel("$RG^1$")
        ax.set_ylabel("$AUROC^2 * PG^1$")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        plt.show()

    def plot_rgrpg_3d(self, fig=None):
        """This method plots the 3d version of the RGPRG surface, with the recall-gains from the
        training data vs reject data classifier on the z-axis and the true positive and false positive rates of the
        corresponding ROC curve of the real training data classifier on y-axis and on the x-axis, respectively.
        The true positive rate (y-axis) of the real training data classifier is multiplied by the precision-gain of the
        training data vs reject data classifier.

        Args:
            None.

        Returns:
            Nothing.

        """

        # Ignore warnings from matplotlib
        warnings.filterwarnings("ignore")
        if fig == None:
            fig = plt.figure()
        ax = fig.gca(projection='3d')

        n_points = np.alen(self.rocs[0])
        crossing_lines = np.zeros((n_points, np.alen(self.recall_gains), 3))

        for i, recall_gain in enumerate(self.recall_gains):
            roc = self.rocs[i]
            ax.plot(roc[:, 1], roc[:, 0] * self.precision_gains[i],
                    np.ones(np.alen(roc))*recall_gain, 'ko-')

            for point in np.arange(n_points):
                crossing_lines[point, i, :] = np.array([roc[point, 1], roc[point, 0] * self.precision_gains[i], recall_gain])

        for point in np.arange(n_points):
            crossing_line = crossing_lines[point]
            ax.plot(crossing_line[:, 0], crossing_line[:, 1], crossing_line[:, 2], 'k-')

        ax.set_xlabel('$FP_r^2$')
        ax.set_ylabel('$TP_r^2 * PG^1$')
        ax.set_zlabel('$RG^1$')

        plt.show()

    def calculate_volume(self):
        """This method calculates the volume under the RGPRG surface.

        Args:
            None.

        Returns:
            float: The volume under the Recall-gain-ROC-Precision-gain surface.

        """
        return auc(self.areas*self.precision_gains, self.recall_gains, reorder=True)


def even_out_roc_points(rocs):
    """This function uses linear interpolation to generate more points for any roc curves with less points than
    the others. This is done to allow for the surface plotting.

    Args:
        rocs (dict{[[float]]}): Roc curves from an RGPRG surface.

    Returns:
        rocs (dict{[[float]]}): The same ROC curves, now with the same number of points for all of them.

    """
    n_recall_gains = np.alen(rocs.keys())
    # Create points for every ROC curve, so that it has the same number of points as the next curve.
    # Start from the second to last curve, since the last one has the highest number of points, and go back to
    # the first one.
    for index in np.arange(n_recall_gains-2, -1, -1):
        next_roc = rocs[index + 1]
        max_points = np.alen(next_roc)
        roc = np.copy(rocs[index])
        n_points_roc = np.alen(roc)
        if n_points_roc < max_points:
            for point in np.arange(1, n_points_roc):
                # Get two points from the roc curve.
                point_1 = roc[point - 1, :]
                point_2 = roc[point, :]
                # Check how many points in the next curve have false positive rates between these two points.
                n_points_interval = np.sum(np.logical_and(next_roc[:, 1] > point_1[1],
                                                          next_roc[:, 1] < point_2[1]))
                if (n_points_interval + n_points_roc) > max_points:
                    n_points_interval = max_points - n_points_roc
                if n_points_interval > 0:
                    new_points = np.zeros((n_points_interval, 2))
                    # Calculate the slope of the line connecting the two points.
                    slope = (point_2[0] - point_1[0]) / (point_2[1] - point_1[1])

                    # Create n_points_interval uniformly between the two points
                    new_points[:, 1] = np.linspace(point_1[1], point_2[1], n_points_interval + 2)[1:-1]
                    new_points[:, 0] = slope * (new_points[:, 1] - point_1[1]) + point_1[0]
                    # Insert the new points between the two selected points
                    rocs[index] = np.insert(rocs[index], point, new_points, 0)
    return rocs
