import numpy as np
from test_optimality import plot_roc_curve
from test_optimality import one_vs_rest_roc_curve

import matplotlib.pyplot as plt
plt.ion()

y = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])
scores = np.array([.05, .15, .16, .18, .2, .2, .45, .55, .7, .7, .7, .85, .9,
                   .9, .95])

pos_label = 0
scores = 1-scores

fig = plt.figure('roc_curve')
ax = fig.add_subplot(111)
plot_roc_curve(y,scores,pos_label=pos_label, fig=fig)

roc = one_vs_rest_roc_curve(y,scores,pos_label)
fig = plt.figure('skew_lines')
fig.clf()
ax = fig.add_subplot(111)
Q_min = roc[0]
Q_max = 1-roc[1]
ax.plot(np.vstack((np.zeros_like(Q_min), np.ones_like(Q_max))),
        np.vstack((Q_min, Q_max)), '--', c='0.80')
ax.set_xlabel('skew')
ax.set_ylabel('$Q_{skew}$')
brier_score = normalized_brier_score(y==pos_label,prediction[:,pos_label])
