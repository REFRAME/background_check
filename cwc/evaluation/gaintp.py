from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import warnings


class GainTP:

    def __init__(self, step1_reject_scores, step1_training_scores, step2_training_scores,
                 training_labels, gain="accuracy", step2_threshold=0.5):

        pos_scores = np.append(np.inf, np.unique(np.append(step1_training_scores, step1_reject_scores))[::-1])
        self.thresholds = np.ones(np.alen(pos_scores)) * -1.0
        self.tpr = np.zeros(np.alen(pos_scores))
        self.gains = np.zeros(np.alen(pos_scores))
        self.gain = gain

        for i, threshold in enumerate(pos_scores):
            n_accepted_rejects = np.sum(step1_reject_scores >= threshold)
            accepted_training = step1_training_scores >= threshold
            new_tpr = np.sum(accepted_training) / np.alen(training_labels)

            if i == 0 or new_tpr != self.tpr[i-1]:
                self.thresholds[i] = threshold
                self.tpr[i] = new_tpr
                self.gains[i] = calculate_gain(step2_training_scores, training_labels,
                                               n_accepted_rejects, accepted_training,
                                               gain=gain, threshold=step2_threshold)
        self.tpr = self.tpr[self.thresholds > -1.0]
        self.gains = self.gains[self.thresholds > -1.0]
        self.thresholds = self.thresholds[self.thresholds > -1.0]

    def plot(self):
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
        plt.scatter(self.tpr, self.gains, c='k', marker='.')
        # index = np.argmax(self.f_gain_betas)
        # plt.scatter(self.recall_gains[index], self.areas[index] * self.precision_gains[index], c='r', marker='o')
        plt.plot(self.tpr, self.gains, 'k-')
        plt.xlabel("$TP_{r1}$")
        plt.ylabel("$" + self.gain + "_2$")
        axes = plt.gca()
        axes.set_xlim([0.0, 1.0])
        axes.set_ylim([0.0, 1.0])
        plt.show()

    def calculate_volume(self):
        """This method calculates the volume under the RGPRG surface.

        Args:
            None.

        Returns:
            float: The volume under the Recall-gain-ROC-Precision-gain surface.

        """
        return auc(self.tpr, self.gains, reorder=True)


def calculate_gain(step2_training_scores, training_labels, n_accepted_rejects, accepted_instances,
                   gain="accuracy", threshold=0.5):
    accepted_scores = step2_training_scores[accepted_instances]
    accepted_labels = training_labels[accepted_instances]
    if gain == "accuracy":
        if np.alen(accepted_labels) == 0:
            return 0
        else:
            n_correct_instances = np.sum(np.logical_not(
                np.logical_xor(accepted_scores >= threshold, accepted_labels == 1)))
            return n_correct_instances / (np.alen(accepted_labels) + n_accepted_rejects)
