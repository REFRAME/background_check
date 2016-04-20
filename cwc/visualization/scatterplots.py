import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA



colors = ['#AAAAAA', '#7777CC', '#FFFF99', '#99FFFF', '#FF5555', '#FF99FF',
          '#5555FF', '#55FF55', '#555555', '#12569A', '#A96521', '#12A965']
shapes = ['s', 's', 'o', '^', 'v', 's', 'o', '^', 'v', 's', 'o', '^']
sizes = [40, 20, 17, 15, 13, 20, 17, 15, 13, 20, 17, 15]
edgecolors = ['#555555', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k']

def plot_predictions(x,p):
    for i in range(p.shape[1]):
        plt.scatter(x[p[:,i],0], x[p[:,i],1], marker=shapes[i], c=colors[i],
                    s=sizes[i], edgecolor=edgecolors[i])
    x_min = np.min(x, axis=0)
    x_max = np.max(x, axis=0)
    plt.xlim([x_min[0], x_max[0]])
    plt.ylim([x_min[1], x_max[1]])

def plot_data_and_reject(x, y, r, max_dim=2, fig=None):
    if fig == None:
        fig = plt.figure('data reject')

    n_features = x.shape[1]
    classes = np.unique(y)
    n_classes = len(classes)

    if n_features > max_dim:
        pc = PCA(n_components=max_dim, whiten=True)
        pc.fit(x)
        r = pc.transform(r)
        x = pc.transform(x)

    if n_features == 1 or max_dim == 1:
        gridspec = GridSpec(1, 4)
        subplotspec = gridspec.new_subplotspec((0, 0), rowspan=1, colspan=3)
        ax = fig.add_subplot(subplotspec)
    elif n_features == 2 or max_dim == 2:
        gridspec = GridSpec(1, 4)
        subplotspec = gridspec.new_subplotspec((0, 0), rowspan=1, colspan=3)
        ax = fig.add_subplot(subplotspec)
    elif n_features >= 3 or max_dim == 3:
        ax = fig.add_subplot(111, projection='3d')

    for i, k in enumerate(classes):
        index = (y == k)
        if n_features == 1 or max_dim == 1:
            ax.hist(x[index,0], label='$C_{}$'.format(i+1))
        elif n_features == 2 or max_dim == 2:
            ax.scatter(x[index,0], x[index,1], marker=shapes[i+1],
                    c=colors[i+1], edgecolor=edgecolors[i+1],
                    label='$C_{}$'.format(i+1))
        elif n_features >= 3 or max_dim == 3:
            ax.scatter(x[index,0], x[index,1], x[index,2], marker=shapes[i+1],
                    c=colors[i+1], edgecolor=edgecolors[i+1],
                    label='$C_{}$'.format(i+1))

    if n_features == 1 or max_dim == 1:
        ax.hist(r[:,0], label='$C_{new}$')
        plt.legend(bbox_to_anchor=(1.05, 0), loc='lower left', borderaxespad=0.)
    elif n_features == 2 or max_dim == 2:
        ax.scatter(r[:,0], r[:,1], marker='+', c=colors[0], label='$C_{new}$')
        x_min = np.min(np.vstack([x.min(axis=0), r.min(axis=0)]), axis=0)
        x_max = np.max(np.vstack([x.max(axis=0), r.max(axis=0)]), axis=0)
        ax.set_xlim([x_min[0], x_max[0]])
        ax.set_ylim([x_min[1], x_max[1]])
        plt.legend(bbox_to_anchor=(1.05, 0), loc='lower left', borderaxespad=0.)
    elif n_features >= 3 or max_dim == 3:
        ax.scatter(r[:,0], r[:,1], r[:,2], marker='+', c=colors[0])

