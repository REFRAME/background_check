from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import warnings


class RGP:
    """This class represents a Recall_1-Gain_2-Precision_1 (RGP) curve. An object of class RGP is built based
    on the result of two models:

    1- The first one is a training data vs reject data classifier and its recall and precision values at
        various thresholds are used to build the curve;
    2- The second (binary) classifier was trained to separate both classes of the original training data. Its
        gain values for all recall values of the first classifier are multiplied by the corresponding precision values
        and used to build the curve.

    Args:
        step1_reject_scores ([float]): Positive scores for the reject data, obtained from
            the training data vs reject data classifier. For each threshold value "theta",
            the classifier accepts an instance "x" when "S_1(x) >= theta", where "S_1(x)" is
            the score given by the first classifier to instance "x".
        step1_training_scores ([float]): Positive scores for the training data, obtained from
            the training data vs reject data classifier. For each threshold value "theta",
            the classifier accepts an instance "x" when "S_1(x) >= theta", where "S_1(x)" is
            the score given by the first classifier to instance "x".
        step2_training_scores ([int]): Positive scores for the training data, obtained from
            the second classifier. For each threshold value "theta",
            the classifier labels an instance "x" as positive when "S_1(x) >= theta", where
            "S_1(x)" is the score given by the second classifier to instance "x".
        training_labels ([int]): Labels of the training data. 1 for the positive class
            and 0 for the negative class.
        gain (str): Which type of gain is used to evaluate the second classifier.
        step2_threshold (float): Threshold used to calculate the gain of the
            second classifier.

    Attributes:
        thresholds ([float]): Thresholds corresponding to the recall and precision values.
        recalls ([float]): Recalls of the first classifier, calculated by thresholding over
        step1_reject_scores and step1_training_scores.
        precisions ([float]): Precisions of the first classifier, calculated by thresholding
        over step1_reject_scores and step1_training_scores.
        gains ([float]): Gain values of the second classifier, calculated using the true
        training instances accepted by the first classifier at the various recall thresholds.
        gain_type (str): Which type of gain is used to evaluate the second classifier.

    """
    def __init__(self, step1_reject_scores, step1_training_scores, step2_training_scores,
                 training_labels, gain="accuracy", step2_threshold=0.5):

        pos_scores = np.append(np.inf, np.unique(np.append(step1_training_scores, step1_reject_scores))[::-1])
        self.thresholds = np.ones(np.alen(pos_scores)) * -1.0
        self.recalls = np.zeros(np.alen(pos_scores))
        self.precisions = np.zeros(np.alen(pos_scores))
        self.gains = np.zeros(np.alen(pos_scores))
        self.gain_type = gain

        for i, threshold in enumerate(pos_scores):
            n_accepted_rejects = np.sum(step1_reject_scores >= threshold)
            accepted_training = step1_training_scores >= threshold
            new_recall = np.sum(accepted_training) / np.alen(training_labels)

            if i == 0 or new_recall != self.recalls[i-1]:
                self.thresholds[i] = threshold
                self.recalls[i] = new_recall
                if np.sum(accepted_training) == 0.0:
                    self.precisions[i] = 0.0
                else:
                    self.precisions[i] = np.sum(accepted_training) / (np.sum(accepted_training) + n_accepted_rejects)
                accepted_scores = step2_training_scores[accepted_training]
                accepted_labels = training_labels[accepted_training]
                self.gains[i] = calculate_gain(accepted_scores, accepted_labels,
                                               gain=gain, threshold=step2_threshold)
        self.recalls = self.recalls[self.thresholds > -1.0]
        self.gains = self.gains[self.thresholds > -1.0]
        self.precisions = self.precisions[self.thresholds > -1.0]
        self.thresholds = self.thresholds[self.thresholds > -1.0]
        pi = np.sum(training_labels == 1) / np.alen(training_labels)
        self.f_betas = calculate_f_betas(self.recalls, self.precisions, self.gains, pi=pi, min_beta=0.5)
        self.values = calculate_values(self.recalls, self.precisions, self.gains)

    def plot(self, fig=None):
        """This method plots the RGP surface, with the recalls from the
        first classifier on the x-axis and the gains of the second classifier,
        multiplied by the corresponding precisions from the first classifier
        on the y-axis. The optimal threshold of the first classifier is shown
        in two ways:

        1- Red dot marks the optimal according to f-beta.
        2- Blue dot marks the optimal according to optimization criterion.

        Args:
            fig (object): An object of a Matplotlib figure
            (as obtained by using Matplotlib's figure() function).

        Returns:
            Nothing.

        """
        # Ignore warnings from matplotlib
        warnings.filterwarnings("ignore")
        if fig is None:
            fig = plt.figure()
        plt.plot(self.recalls, self.gains * self.precisions, 'k.-')
        index = np.argmax(self.f_betas)
        plt.plot(self.recalls[index], self.gains[index] * self.precisions[index], 'ro')
        index = np.argmax(self.values)
        plt.plot(self.recalls[index], self.gains[index] * self.precisions[index], 'bo')
        plt.xlabel("$Recall_1$")
        plt.ylabel("$" + self.gain_type + "'_2$")
        axes = plt.gca()
        axes.set_xlim([0.0, 1.0])
        axes.set_ylim([0.0, 1.0])
        plt.show()

    def get_optimal_step1_threshold(self):
        """This method returns the threshold with the highest f_beta on the RGP curve.

        Args:
            None.

        Returns:
            float: The threshold with the highest f_beta on the RGP curve.

        """
        return self.thresholds[np.argmax(self.f_betas)]

    def calculate_area(self):
        """This method calculates the area under the RGP curve,
        by invoking Scikit Learn's auc function, which calculates
        the area using the trapezoid rule.

        Args:
            None.

        Returns:
            float: The volume under the Recall-gain-ROC-Precision-gain surface.

        """
        return auc(self.recalls, self.gains*self.precisions, reorder=True)


def calculate_gain(accepted_scores, accepted_labels, gain="accuracy", threshold=0.5):
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
    if gain == "accuracy":
        if np.alen(accepted_labels) == 0:
            return 0
        else:
            n_correct_instances = np.sum(np.logical_not(
                np.logical_xor(accepted_scores >= threshold, accepted_labels == 1)))
            return n_correct_instances / np.alen(accepted_labels)


def calculate_f_betas(recalls, precisions, gains, pi=0.5, min_beta=0.5):
    """This function calculates f-beta values based on the given recall and precision values.
        The beta value is taken based on the gain_2 value corresponding to recall_1 = 1,
        where gain_2 is the gain value of the second classifier and recall_1 is the
        recall value of the first classifier. At recall_1 = 1, the first classifier accepts
        all true training data. If gain_2 is the second classifier's accuracy, we have 2
        extreme scenarios:

        1- The second classifier has accuracy_2 = 1.0 at recall_1 = 1: this means that the
        second classifier is perfectly accurate on the complete training data set. Therefore,
        it is also perfect with any subset of the true training data that gets accepted by the
        first classifier. Thus, recall_1 is not very important and only precision_1 is able
        to impact the performance of the second classifier. In this scenario, beta should be
        chosen as min_beta, since lower beta values give more weight to precision.

        2- The second classifier has the worst possible binary classification
        performance (accuracy_2 = pi, where pi is the proportion of positives) at recall_1 = 1:
        Here, precision_1 and recall_1 are equally (un)important, since the second classifier
        is the main responsible for the poor aggregated performance. Therefore, beta should be
        chosen as 1, giving similar weights to recall_1 and precision_1.

        Args:
            recalls ([float]): Recalls of the first classifier.
            precisions ([float]): Precisions of the first classifier.
            gains ([float]): Gains of the second classifier.
            pi (float): Proportion of positives in the true training data.
            min_beta (float): Minimum value of beta.

        Returns:
            float: The calculated f_betas.

    """
    warnings.filterwarnings("ignore")
    a = (1.0 - min_beta) / (pi - 1.0)
    beta = a * gains[np.argmax(recalls)] + min_beta - a
    f_betas = (1 + beta**2.0) * ((precisions * recalls) / (beta**2.0 * precisions + recalls))
    f_betas[np.isnan(f_betas)] = 0.0
    return f_betas


def calculate_values(recalls, precisions, gains):
    """This function calculates the optimization value corresponding
     to each operating point of the aggregated classifiers.


        Args:
            recalls ([float]): Recalls of the first classifier.
            precisions ([float]): Precisions of the first classifier.
            gains ([float]): Gains of the second classifier.

        Returns:
            float: The calculated values.

    """
    values = gains * precisions - (np.abs(precisions - recalls) / (precisions+recalls))
    values[np.isnan(values)] = np.amin(values[np.logical_not(np.isnan(values))]) - 1
    return values
