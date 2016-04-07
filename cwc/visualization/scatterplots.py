import numpy as np
import matplotlib.pyplot as plt

colors = ['#AAAAAA', '#7777CC', '#FFFF99', '#99FFFF', '#FF5555']
shapes = ['s', 's', 'o', '^', 'v']
sizes = [40, 20, 17, 15, 13]
edgecolors = ['#555555', 'k', 'k', 'k', 'k']

def plot_predictions(x,p):
    for i in range(p.shape[1]):
        plt.scatter(x[p[:,i],0], x[p[:,i],1], marker=shapes[i], c=colors[i],
                    s=sizes[i], edgecolor=edgecolors[i])
    x_min = np.min(x, axis=0)
    x_max = np.max(x, axis=0)
    plt.xlim([x_min[0], x_max[0]])
    plt.ylim([x_min[1], x_max[1]])

def plot_data_and_reject(x, y, r, fig=None):
    if fig == None:
        fig = plt.figure('data reject')

    n_features = x.shape[1]
    classes = np.unique(y)
    n_classes = len(classes)

    if n_features >= 3:
        ax = fig.add_subplot(111, projection='3d')
    elif n_features == 2:
        ax = fig.add_subplot(111)
    elif n_features == 1:
        ax = fig.add_subplot(111)

    for i, k in enumerate(classes):
        index = (y == k)
        if n_features >= 3:
            ax.scatter(x[index,0], x[index,1], x[index,2], marker=shapes[i+1],
                    c=colors[i+1], edgecolor=edgecolors[i+1])
        elif n_features == 2:
            ax.scatter(x[index,0], x[index,1], marker=shapes[i+1],
                    c=colors[i+1], edgecolor=edgecolors[i+1])
        elif n_features == 1:
            ax.hist(x[index,0])

    if n_features >= 3:
        ax.scatter(r[:,0], r[:,1], r[:,2], marker='+', c=colors[0])
    elif n_features == 2:
        ax.scatter(r[:,0], r[:,1], marker='+', c=colors[0])
    elif n_features == 1:
        ax.hist(r[:,0])
