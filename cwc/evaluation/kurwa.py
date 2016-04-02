from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import warnings


class Kurwa:
    """This class represents a Recall_1-Gain_2-Precision_1 (RGP) curve.

    An object of class RGP is built based on the result of two models:

    1- The first one is a training data vs reject data classifier and its
      recall and precision values at various thresholds are used to build the
      curve;
    2- The second (binary) classifier was trained to separate both classes of
      the original training data. Its gain values for all recall values of the
      first classifier are multiplied by the corresponding precision values and
      used to build the curve.

    Args:
        step1_reject_scores ([float]): Positive scores for the reject data,
          obtained from the training data vs reject data classifier. For each
          threshold value "theta", the classifier accepts an instance "x" when
          "S_1(x) >= theta", where "S_1(x)" is the score given by the first
          classifier to instance "x".
        step1_training_scores ([float]): Positive scores for the training data,
          obtained from the training data vs reject data classifier. For each
          threshold value "theta", the classifier accepts an instance "x" when
          "S_1(x) >= theta", where "S_1(x)" is the score given by the first
          classifier to instance "x".
        step2_training_scores ([int]): Positive scores for the training data,
          obtained from the second classifier. For each threshold value "theta",
          the classifier labels an instance "x" as positive when "S_1(x) >=
          theta", where "S_1(x)" is the score given by the second classifier to
          instance "x".
        training_labels ([int]): Labels of the training data. 1 for the
          positive class and 0 for the negative class.
        gain (str): Which type of gain is used to evaluate the second classifier.
        step2_threshold (float): Threshold used to calculate the gain of the
            second classifier.

    Attributes:
        thresholds ([float]): Thresholds corresponding to the recall and
          precision values.
        recalls ([float]): Recalls of the first classifier, calculated by
          thresholding over step1_reject_scores and step1_training_scores.
        precisions ([float]): Precisions of the first classifier, calculated by
          thresholding over step1_reject_scores and step1_training_scores.
        gains ([float]): Gain values of the second classifier, calculated using
          the true training instances accepted by the first classifier at the
          various recall thresholds.
        gain_type (str): Which type of gain is used to evaluate the second
          classifier.
        positive_proportion (float): the proportion of positives (true training
          data) scored by the first classifier.

    """
    def __init__(self, step1_reject_scores, step1_training_scores,
                 step2_training_scores, training_labels):

        self.thresholds = np.linspace(1.0, 0.0, 100)
        self.recalls = np.zeros(np.alen(self.thresholds))
        self.precisions = np.zeros(np.alen(self.thresholds))
        self.accuracies = np.zeros(np.alen(self.thresholds))

        for i, threshold in enumerate(self.thresholds):
            n_accepted_rejects = np.sum(step1_reject_scores >= threshold)
            accepted_training = step1_training_scores >= threshold
            new_recall = np.sum(accepted_training) / np.alen(training_labels)

            self.recalls[i] = new_recall
            sum_at = np.sum(accepted_training)
            if (sum_at + n_accepted_rejects) == 0.0:
                self.precisions[i] = np.nan
            else:
                self.precisions[i] = sum_at / (sum_at + n_accepted_rejects)
            accepted_scores = step2_training_scores[accepted_training]
            accepted_labels = training_labels[accepted_training]
            self.accuracies[i] = calculate_accuracy(accepted_scores,
                                                    accepted_labels)

    def plot(self, fig=None):
        """This method plots the RGP surface, with the recalls from the
        first classifier on the x-axis and the gains of the second classifier,
        multiplied by the corresponding precisions from the first classifier
        on the y-axis. The optimal threshold of the first classifier is shown
        in two ways:

        1- Black circle marks the optimal according to f-beta.
        2- Red dot marks the optimal according to optimization criterion.

        Args:
            fig (object): An object of a Matplotlib figure
                (as obtained by using Matplotlib's figure() function).
            baseline (bool): True means that the baseline will be drawn.
                The baseline is built by taking the worst precision
                (proportion of positives) for every recall value.
        Returns:
            Nothing.

        """
        # Ignore warnings from matplotlib
        warnings.filterwarnings("ignore")
        if fig is None:
            fig = plt.figure()
        plt.plot(self.recalls, self.accuracies, 'k.-')
        plt.xlabel("$\mathrm{Recall}_1$")
        plt.ylabel("$\mathrm{Accuracy}_2$")
        axes = plt.gca()
        axes.set_xlim([0.0, 1.01])
        axes.set_ylim([0.0, 1.0])
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        axes.get_xaxis().tick_bottom()
        axes.get_yaxis().tick_left()
        plt.show()

    @property
    def accuracies(self):
        return self.accuracies

    @property
    def recalls(self):
        return self.recalls

def calculate_accuracy(accepted_scores, accepted_labels):
    """This function calculates the gain of the second classifier, based on the true
    training instances accepted by the first classifier.

        Args:
            accepted_scores ([int]): Positive scores obtained from
                the second classifier for the true training data
                accepted by the first classifier. For each threshold value "theta",
                the second classifier labels an instance "x" as positive when "S_1(x) >= theta", where
                "S_1(x)" is the score given by the second classifier to instance "x".
            accepted_labels ([int]): Labels of the true training data
                accepted by the first classifier. 1 for the positive class
                and 0 for the negative class.
            gain (str): Which type of gain is used to evaluate the second classifier.
            threshold (float): Threshold used to calculate the gain of the
                second classifier.

        Returns:
            float: The gain of the second classifier, based on the true
                training instances accepted by the first classifier.

    """
    if np.alen(accepted_labels) == 0:
        return 0
    else:
        return np.mean(np.argmax(accepted_scores, axis=1) == accepted_labels)
