import numpy as np
from cwc.evaluation.metrics import one_vs_rest_roc_curve
from cwc.visualization.cost_lines import plot_skew_lines
from cwc.visualization.roc_analysis import plot_roc_curve

import matplotlib.pyplot as plt
plt.ion()
plt.rcParams['figure.autolayout'] = True

if __name__ == '__main__':
    y = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])
    scores = np.array([.05, .15, .16, .18, .2, .2, .45, .55, .7, .7, .7, .85, .9,
                       .9, .95])
    pos_label = 0
    scores = 1-scores

    fig = plt.figure('roc_curve')
    ax = fig.add_subplot(111)
    plot_roc_curve(y,scores,pos_label=pos_label, fig=fig)

    plot_skew_lines(y,scores,pos_label=pos_label, lower_envelope=True)

    roc = one_vs_rest_roc_curve(y,scores,pos_label)
    fpr = roc[0]
    tpr = roc[1]
    thresholds = roc[2]
    q_skew = thresholds*(1-tpr) + (1-thresholds)*fpr
    # look at the cost function
    q_cost = thresholds*(1-tpr) + (1-thresholds)*fpr
    ax.plot(thresholds, q_cost, 'go-')


    T = np.column_stack([thresholds, thresholds]).flatten()
