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
        step1_thresholds ([float]): Thresholds corresponding to the recall-gain and precision-gain values.
        step2_rocs (dict{[[float]]}): The ROC curves calculated by thresholding over step2_scores and using step2_labels.
        step2_thresholds (dict{[[float]]}): Thresholds used to build the ROC curves.
        areas ([float]): The areas of the roc curves calculated by thresholding over step2_scores and using
            step2_labels.

    """
    def __init__(self, step1_reject_scores, step1_training_scores, step2_training_scores, training_labels):
        self.step1_scores = np.append(step1_training_scores, step1_reject_scores)
        self.step1_labels = np.append(np.ones(np.alen(training_labels)), np.zeros(np.alen(step1_reject_scores)))
        self.step2_scores = step2_training_scores
        self.step2_labels = training_labels

        prg_curve = calculate_prg_points(self.step1_labels, self.step1_scores)
        self.recall_gains = prg_curve['recall_gain'][prg_curve['recall_gain'] >= 0]
        self.precision_gains = prg_curve['precision_gain'][prg_curve['recall_gain'] >= 0]
        self.step1_thresholds = prg_curve['pos_score'][prg_curve['recall_gain'] >= 0]

        n_recalls = np.alen(self.recall_gains)
        self.areas = np.zeros(n_recalls)

        self.step2_rocs = dict()
        self.step2_thresholds = dict()
        self.accuracies = np.zeros(n_recalls)

        for rg in np.arange(n_recalls):
            true_positive_indices = np.where(np.logical_and(self.step1_scores >= self.step1_thresholds[rg],
                                                            self.step1_labels == 1))[0]
            probabilities = step2_training_scores[true_positive_indices]
            labels = training_labels[true_positive_indices]

            threshold = 0.5
            n_correct_instances = np.sum(np.logical_not(
                np.logical_xor(probabilities >= threshold,
                               labels == 1)))
            self.accuracies[rg] = n_correct_instances / np.alen(labels)

            #self.areas[rg] = roc_auc_score(labels, probabilities)
            #fpr, tpr, self.step2_thresholds[rg] = roc_curve(labels, probabilities)
            #self.step2_rocs[rg] = np.append(tpr.reshape(-1, 1), fpr.reshape(-1, 1), axis=1)

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
        if fig is None:
            fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.recall_gains, self.areas*self.precision_gains, 'k.-')
        ax.set_xlabel("$RG^1$")
        ax.set_ylabel("$AUROC^2 * PG^1$")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        plt.show()

    def plot_simple_3d(self, fig=None):
        warnings.filterwarnings("ignore")
        if fig is None:
            fig = plt.figure()

        ax = fig.gca(projection='3d')
        ax.plot(self.recall_gains, self.precision_gains, self.accuracies, 'k.-')
        ax.set_xlabel('$RG_1$')
        ax.set_ylabel('$PG_1$')
        ax.set_zlabel('$ACC_2$')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_zlim([0.0, 1.0])
        plt.show()

    def plot_rgrpg_3d(self, n_recalls=10, n_points_roc=10, fig=None):
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
        warnings.filterwarnings("ignore")
        if fig is None:
            fig = plt.figure()

        [x, y, z] = self.generate_grid(n_recalls=n_recalls, n_points_roc=n_points_roc)

        ax = fig.gca(projection='3d')

        generated_recalls = np.unique(z)
        crossing_lines = np.zeros((n_points_roc, n_recalls, 3))

        for i, recall_gain in enumerate(generated_recalls):
            x_i = x[z == recall_gain]
            y_i = y[z == recall_gain]
            z_i = z[z == recall_gain]
            ax.plot(x_i, y_i, z_i, 'k.-')

            for point in np.arange(n_points_roc):
                crossing_lines[point, i, :] = np.array([x_i[point], y_i[point], z_i[point]])

        for point in np.arange(n_points_roc):
            crossing_line = crossing_lines[point]
            ax.plot(crossing_line[:, 0], crossing_line[:, 1], crossing_line[:, 2], 'k-')
        ax.view_init(elev=10., azim=150)
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

    def generate_grid(self, n_recalls=10, n_points_roc=10):
        [generated_recalls, generated_precisions, generated_thresholds] = \
            uniform_curve(self.recall_gains, self.precision_gains, n_recalls, thresholds=self.step1_thresholds)
        x = np.zeros(n_recalls*n_points_roc)
        y = np.zeros(n_recalls*n_points_roc)
        z = np.zeros(n_recalls*n_points_roc)
        insert_point = 0
        for i, threshold in enumerate(generated_thresholds):
            true_positive_indices = np.where(np.logical_and(self.step1_scores >= threshold,
                                                            self.step1_labels == 1))[0]
            probabilities = self.step2_scores[true_positive_indices]
            labels = self.step2_labels[true_positive_indices]
            fpr, tpr, thresholds = roc_curve(labels, probabilities)
            [fpr, tpr, thresholds] = uniform_curve(fpr, tpr, n_points_roc)
            x[insert_point:insert_point + n_points_roc] = fpr
            y[insert_point:insert_point + n_points_roc] = tpr * generated_precisions[i]
            z[insert_point:insert_point + n_points_roc] = generated_recalls[i]
            insert_point += n_points_roc
        return [x, y, z]


def uniform_curve(x, y, n_points, thresholds=[]):
    generated_x = np.linspace(0.0, 1.0, n_points)
    generated_y = np.zeros(n_points)
    generated_thresholds = np.zeros(n_points)

    insert_point = 0

    index_1 = 0
    index_2 = 1
    while insert_point < n_points:
        x_1 = x[index_1]
        x_2 = x[index_2]
        y_1 = y[index_1]
        y_2 = y[index_2]

        points_interval = np.logical_and(generated_x >= x_1,
                                         generated_x < x_2)
        if index_2 == np.alen(x) - 1:
            points_interval = np.logical_and(generated_x >= x_1,
                                             generated_x <= x_2)
        n_points_interval = np.sum(points_interval)
        if n_points_interval > 0:
            # Calculate the slope of the line connecting the two points.
            slope = (y_2 - y_1) / (x_2 - x_1)
            generated_y[insert_point:insert_point + n_points_interval] = \
                slope * (generated_x[points_interval] - x_1) + y_1
            if np.alen(thresholds) > 0:
                threshold_1 = thresholds[index_1]
                threshold_2 = thresholds[index_2]

                new_thresholds = (threshold_2 - threshold_1) / \
                                 (np.divide((x_2 - x_1), generated_x[points_interval]))
                generated_thresholds[insert_point:insert_point + n_points_interval] = new_thresholds
            insert_point = insert_point + n_points_interval
            index_1 = index_2
        index_2 += 1
    return [generated_x, generated_y, generated_thresholds]
