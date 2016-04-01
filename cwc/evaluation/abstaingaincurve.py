from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import warnings


class AbstainGainCurve:
    """This class represents an Abstain-gain (AG) curve. An object of class
    AbstainGainCurve is built based on the result of two models:

    1- The first one is a training data vs reject data classifier and its
        recall and precision values at various thresholds are used to build
        the curve;
    2- The second (binary) classifier was trained to separate both classes of
        the original training data. Its gain values for all recall values of
        the first classifier are multiplied by the corresponding precision
        values and used to build the curve.

    Args:
        step1_reject_scores ([float]): Positive scores for the reject data,
            obtained from the training data vs reject data classifier. For
            each threshold value "theta", the classifier accepts an instance
            "x" when "S_1(x) >= theta", where "S_1(x)" is the score given by
            the first classifier to instance "x".
        step1_training_scores ([float]): Positive scores for the training
            data, obtained from the training data vs reject data classifier.
            For each threshold value "theta", the classifier accepts an
            instance "x" when "S_1(x) >= theta", where "S_1(x)" is the score
            given by the first classifier to instance "x".
        step2_training_scores ([int]): Positive scores for the training data,
            obtained from the second classifier. For each threshold value
            "theta", the classifier labels an instance "x" as positive when
            "S_1(x) >= theta", where "S_1(x)" is the score given by the
            second classifier to instance "x".
        training_labels ([int]): Labels of the training data. 1 for the
            positive class and 0 for the negative class.
        gain (str): Which type of gain is used to evaluate the second
            classifier.
        step2_threshold (float): Threshold used to calculate the gain of the
            second classifier.

    Attributes:
        thresholds ([float]): Thresholds corresponding to the recall and
            precision values.
        recalls ([float]): Recalls of the first classifier, calculated by
            thresholding over step1_reject_scores and step1_training_scores.
        precisions ([float]): Precisions of the first classifier, calculated
            by thresholding over step1_reject_scores and step1_training_scores.
        gains ([float]): Gain values of the second classifier, calculated
            using the true training instances accepted by the first
            classifier at the various recall thresholds.
        gain_type (str): Which type of gain is used to evaluate the second
            classifier.
        positive_proportion (float): the proportion of positives (true
            training data) scored by the first classifier.

    """
    def __init__(self, step1_reject_scores, step1_training_scores,
                 step2_training_scores,
                 training_labels, gain="accuracy", step2_threshold=0.5):

        pos_scores = np.append(np.inf, np.unique(np.append(
            step1_training_scores, step1_reject_scores))[::-1])
        self.thresholds = np.ones(np.alen(pos_scores)) * -1.0
        self.recalls = np.zeros(np.alen(pos_scores))
        self.precisions = np.zeros(np.alen(pos_scores))
        self.gains = np.zeros(np.alen(pos_scores))
        self.gain_type = gain
        self.positive_proportion = np.alen(step1_training_scores) / (np.alen(
            step1_training_scores) + np.alen(step1_reject_scores))
        self.mod_gains_ag = np.zeros(np.alen(pos_scores))
        self.recalls_ag = np.zeros(np.alen(pos_scores))
        all_accepted_training = np.zeros((np.alen(pos_scores),  np.alen(
            training_labels)))
        g = calculate_gain(step2_training_scores, training_labels, gain=gain,
                           threshold=step2_threshold)

        for i, threshold in enumerate(pos_scores):
            n_accepted_rejects = np.sum(step1_reject_scores >= threshold)
            accepted_training = step1_training_scores >= threshold
            new_recall = np.sum(accepted_training) / np.alen(training_labels)

            if i == 0 or new_recall != self.recalls[i-1]:
                self.thresholds[i] = threshold
                self.recalls[i] = new_recall
                all_accepted_training[i] = accepted_training
                if (np.sum(accepted_training) + n_accepted_rejects) == 0.0:
                    self.precisions[i] = np.nan
                else:
                    self.precisions[i] = np.sum(accepted_training) / (
                        np.sum(accepted_training) + n_accepted_rejects)
                accepted_scores = step2_training_scores[accepted_training]
                accepted_labels = training_labels[accepted_training]
                self.gains[i] = calculate_gain(accepted_scores, accepted_labels,
                                               gain=gain,
                                               threshold=step2_threshold)

                denominator = (1.0 - g * self.positive_proportion)
                self.mod_gains_ag[i] = (self.gains[i]*self.precisions[i] -
                                     g*self.positive_proportion) / denominator
                self.recalls_ag[i] = (self.recalls[i] -
                                   g*self.positive_proportion) / denominator
        self.recalls = self.recalls[self.thresholds > -1.0]
        self.gains = self.gains[self.thresholds > -1.0]
        self.precisions = self.precisions[self.thresholds > -1.0]
        self.mod_gains_ag = self.mod_gains_ag[self.thresholds > -1.0]
        self.recalls_ag = self.recalls_ag[self.thresholds > -1.0]
        all_accepted_training = all_accepted_training[self.thresholds > -1.0]
        self.thresholds = self.thresholds[self.thresholds > -1.0]
        [self.recalls_ag, self.mod_gains_ag] = generate_crossing_points(
            all_accepted_training.astype(bool),
                                 step1_reject_scores, step1_training_scores,
                                 step2_training_scores,
                                 training_labels, self.recalls_ag,
                                 self.mod_gains_ag,
                                 g * self.positive_proportion,
                                 gain=self.gain_type,
                                 threshold=step2_threshold)

        self.mod_gains_ag = self.mod_gains_ag[self.recalls_ag >= 0]
        self.recalls_ag = self.recalls_ag[self.recalls_ag >= 0]

    def plot(self, fig=None, baseline=True):
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
        plt.plot(self.recalls_ag, self.mod_gains_ag, 'k.-')
        plt.xlabel("$Recall-AG_1$")
        plt.ylabel("$acc-AG_2$")
        axes = plt.gca()
        axes.set_xlim([0.0, 1.01])
        axes.set_ylim([0.0, 1.0])
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        axes.get_xaxis().tick_bottom()
        axes.get_yaxis().tick_left()
        plt.show()

    def calculate_area(self):
        """This method calculates the area under the RGP curve,
        by invoking Scikit Learn's auc function, which calculates
        the area using the trapezoid rule.

        Args:
            None.

        Returns:
            float: The volume under the Recall-gain-ROC-Precision-gain surface.

        """
        return auc(self.recalls_ag, self.mod_gains_ag, reorder=True)


def calculate_gain(accepted_scores, accepted_labels, gain="accuracy",
                   threshold=0.5):
    """This function calculates the gain of the second classifier, based on the
    true training instances accepted by the first classifier.

        Args:
            accepted_scores ([int]): Positive scores obtained from
                the second classifier for the true training data
                accepted by the first classifier. For each threshold value
                "theta", the second classifier labels an instance "x" as
                positive when "S_1(x) >= theta", where "S_1(x)" is the score
                given by the second classifier to instance "x".
            accepted_labels ([int]): Labels of the true training data
                accepted by the first classifier. 1 for the positive class
                and 0 for the negative class.
            gain (str): Which type of gain is used to evaluate the second
                classifier.
            threshold (float): Threshold used to calculate the gain of the
                second classifier.

        Returns:
            float: The gain of the second classifier, based on the true
                training instances accepted by the first classifier.

    """
    if gain == "accuracy":
        if np.alen(accepted_labels) == 0:
            return np.nan
        else:
            n_correct_instances = np.sum(np.logical_not(
                np.logical_xor(accepted_scores >= threshold,
                               accepted_labels == 1)))
            return n_correct_instances / np.alen(accepted_labels)


def generate_crossing_points(all_accepted_training, step1_reject_scores,
                             step1_training_scores, step2_training_scores,
                             training_labels,
                             recalls_ag, mod_gains_ag,
                             baseline, gain="accuracy", threshold=0.5):

    non_negative_indices = np.where(recalls_ag >= 0)[0]

    j = np.amin(non_negative_indices)
    if recalls_ag[j] > 0:
        rag = recalls_ag[j]
        true_positives = all_accepted_training[j]
        while rag > 0:
            min_accepted_score = np.amin(step1_training_scores[true_positives])
            index = np.random.choice(np.where(step1_training_scores ==
                                     min_accepted_score)[0], 1)
            true_positives[index] = False
            new_recall = np.sum(true_positives) / np.alen(training_labels)
            rag = (new_recall - baseline) / (1.0 - baseline)
            accepted_rejects = step1_reject_scores >= min_accepted_score
            new_precision = np.sum(true_positives) / (
                np.sum(true_positives) + np.sum(accepted_rejects))
        new_gain = calculate_gain(step2_training_scores[true_positives],
                                  training_labels[true_positives],
                                               gain=gain,
                                               threshold=threshold)
        new_mod_gain_ag = (new_gain * new_precision - baseline) / (1 - baseline)
        recalls_ag = np.insert(recalls_ag, j, 0)
        mod_gains_ag = np.insert(mod_gains_ag, j, new_mod_gain_ag)

    # Add final point in the right-hand side of the curve
    min_mod_gain_ag = np.amin(mod_gains_ag[non_negative_indices])
    if min_mod_gain_ag > 0:
        recalls_ag = np.append(recalls_ag, 1)
        mod_gains_ag = np.append(mod_gains_ag, 0)
    return [recalls_ag, mod_gains_ag]
