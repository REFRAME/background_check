import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

cm_ra = {'red': ((0.0, 1.0, 1.0),
                 (1.0, 1.0, 1.0)),
        'green': ((0.0, 1.0, 1.0),
                  (1.0, 0.0, 0.0)),
        'blue': ((0.0, 1.0, 1.0),
                  (1.0, 0.0, 0.0)),
        'alpha': ((0.0, 0.0, 0.0),
                  (1.0, 1.0, 1.0))}
red_alpha = LinearSegmentedColormap('RedAlpha1', cm_ra)
cm_ba = {'red': ((0.0, 1.0, 1.0),
                 (1.0, 0.0, 0.0)),
        'green': ((0.0, 1.0, 1.0),
                  (1.0, 0.0, 0.0)),
        'blue': ((0.0, 1.0, 1.0),
                  (1.0, 1.0, 1.0)),
        'alpha': ((0.0, 0.0, 0.0),
                  (1.0, 1.0, 1.0))}
blue_alpha = LinearSegmentedColormap('BlueAlpha1', cm_ba)
cm_ya = {'red': ((0.0, 1.0, 1.0),
                 (1.0, 1.0, 1.0)),
        'green': ((0.0, 1.0, 1.0),
                  (1.0, 1.0, 1.0)),
        'blue': ((0.0, 1.0, 1.0),
                  (1.0, 0.0, 0.0)),
        'alpha': ((0.0, 0.0, 0.0),
                  (1.0, 1.0, 1.0))}
yellow_alpha = LinearSegmentedColormap('YellowAlpha1', cm_ya)
cm_ga = {'red': ((0.0, 1.0, 1.0),
                 (1.0, 0.0, 0.0)),
        'green': ((0.0, 1.0, 1.0),
                  (1.0, 1.0, 1.0)),
        'blue': ((0.0, 1.0, 1.0),
                  (1.0, 0.0, 0.0)),
        'alpha': ((0.0, 0.0, 0.0),
                  (1.0, 1.0, 1.0))}
green_alpha = LinearSegmentedColormap('GreenAlpha1', cm_ga)
cm_ca = {'red': ((0.0, 1.0, 1.0),
                 (1.0, 0.0, 0.0)),
        'green': ((0.0, 1.0, 1.0),
                  (1.0, 1.0, 1.0)),
        'blue': ((0.0, 1.0, 1.0),
                  (1.0, 1.0, 1.0)),
        'alpha': ((0.0, 0.0, 0.0),
                  (1.0, 1.0, 1.0))}
cyan_alpha = LinearSegmentedColormap('CyanAlpha1', cm_ca)

cmaps = [plt.cm.Greys, blue_alpha, yellow_alpha, cyan_alpha, red_alpha]

def plot_probabilities(p, alpha=1.0):
    delta = np.sqrt(p.shape[0])
    for i in range(p.shape[1]):
        plt.imshow(p[:,i].reshape((delta,delta)), cmap=cmaps[i], alpha=alpha)
#       plt.pcolormesh(MX1, MX2, q_grid[:,2].reshape((delta,delta)), alpha=0.5,
#                      cmap=plt.cm.YlOrBr)
    plt.ylim([0,delta-1])
#   plt.xlim([x_min[0], x_max[0]])
#   plt.ylim([x_min[1], x_max[1]])
