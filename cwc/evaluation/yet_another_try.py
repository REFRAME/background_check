from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from prg import calculate_prg_points
from sklearn.metrics import auc
import warnings


class YetAnotherTry:
    def __init__(self, step1_reject_scores, step1_training_scores,
                 step2_training_scores, training_labels):
        step1_scores = np.append(step1_training_scores,
                                 step1_reject_scores)
        step1_labels = np.append(np.ones(np.alen(training_labels)),
                                 np.zeros(np.alen(step1_reject_scores)))
        prg_curve = calculate_prg_points(step1_labels, step1_scores)
        self.recall_gains = prg_curve['recall_gain'][prg_curve[
                                                         'recall_gain'] >= 0]
        self.precision_gains = prg_curve[
            'precision_gain'][prg_curve['recall_gain'] >= 0]
        self.step1_thresholds = prg_curve[
            'pos_score'][prg_curve['recall_gain'] >= 0]
        n_recalls = np.alen(self.recall_gains)
        self.areas = np.zeros(n_recalls)
        self.step2_rocs = dict()
        self.build_penalized_roc_curves(step1_reject_scores,
                                        step1_training_scores,
                                        step2_training_scores,
                                        training_labels)
        self.calculate_areas()

    def plot_2d(self, fig=None):
        warnings.filterwarnings("ignore")
        if fig is None:
            fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.recall_gains, self.areas, 'k.-')
        ax.set_xlabel("$RG^1$")
        ax.set_ylabel("$AUPROC^2$")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        plt.show()

    def build_penalized_roc_curves(self, step1_reject_scores,
                                   step1_training_scores, step2_training_scores,
                                   training_labels):
        n_recalls = np.alen(self.recall_gains)
        for rg in np.arange(n_recalls):
            tp1_indices = np.where(step1_training_scores >=
                                   self.step1_thresholds[rg])[0]
            fp1 = np.sum(step1_reject_scores >= self.step1_thresholds[rg])
            probabilities = step2_training_scores[tp1_indices]
            labels = training_labels[tp1_indices]
            pos_scores = np.append(np.inf, np.unique(probabilities)[::-1])
            if np.amin(pos_scores) > 0.0:
                pos_scores = np.append(pos_scores, 0.0)
            thresholds = np.ones(np.alen(pos_scores)) * -1.0
            tpr = np.zeros(np.alen(thresholds))
            fpr = np.zeros(np.alen(thresholds))

            previous_tpr2 = previous_fpr2 = 0.0

            for i, threshold in enumerate(pos_scores):
                tp2 = np.sum(np.logical_and(probabilities >= threshold,
                             labels == 1))
                fn2 = np.sum(np.logical_and(probabilities < threshold,
                             labels == 1)) + fp1
                fp2 = np.sum(np.logical_and(probabilities >= threshold,
                             labels == 0)) + fp1
                tn2 = np.sum(np.logical_and(probabilities < threshold,
                             labels == 0))
                new_tpr2 = tp2 / (tp2 + fn2)
                new_fpr2 = fp2 / (fp2 + tn2)
                if i == 0 or new_tpr2 != previous_tpr2 \
                        or new_fpr2 != previous_fpr2:
                    thresholds[i] = threshold
                    tpr[i] = new_tpr2
                    fpr[i] = new_fpr2
            tpr = tpr[thresholds > -1.0]
            fpr = fpr[thresholds > -1.0]
            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            # ax.plot(fpr, tpr, 'k.-')
            #
            # ax.plot([fpr[0],fpr[-1]], [tpr[0],tpr[-1]], '--')
            # ax.set_xlabel("$fpr$")
            # ax.set_ylabel("$tpr$")
            # ax.set_xlim([0.0, 1.0])
            # ax.set_ylim([0.0, 1.0])
            # plt.show()
            self.step2_rocs[rg] = np.append(tpr.reshape(-1, 1),
                                            fpr.reshape(-1, 1), axis=1)

    def calculate_areas(self):
        n_recalls = np.alen(self.recall_gains)
        for rg in np.arange(n_recalls):
            roc = self.step2_rocs[rg]
            self.areas[rg] = auc(roc[:, 1], roc[:, 0], reorder=True)