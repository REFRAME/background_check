import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def plot_target_distributions(datasets, key=None, fig=None):
    if fig is None:
        fig = plt.figure('target_distributions')
    fig.clf()

    width = 0.70

    if key:
        classes, counts = np.unique(targets, return_counts=True)
        plt.bar(classes, counts, width)
        plt.xticks(classes+width/2., datasets[key].names)
        plt.title(key)
    else:
        targets = {}
        for key, value in datasets.iteritems():
            targets[key] = value.target
        square = ceil(sqrt(len(targets)))
        for i, (key, value) in enumerate(targets.iteritems()):
            print('Dataset = {}'.format(key))
            classes, counts = np.unique(value, return_counts=True)
            print('Classes = {}'.format(classes))
            print('Counts = {}'.format(counts))
            plt.subplot(square, square, i+1)
            plt.bar(classes, counts, width)
            plt.xticks(classes+width/2., datasets[key].names)
            plt.title(key)
