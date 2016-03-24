from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import warnings


class ConfidenceInterval:
    def __init__(self, values, x_labels, n_samples=1000, alpha=0.05):
        n_intervals = np.alen(x_labels)
        sample_size = np.shape(values, 1)
        self.intervals = np.zeros(n_intervals, 2)
        self.labels = x_labels
        for interval in np.arange(n_intervals):
            v = values[:, interval]
            means = np.zeros(n_samples,1)
    
            for i in np.arange(n_samples):
                sample = np.random.choice(v, sample_size)
                means[i] = np.mean(sample)
            
            sorted_means = np.sort(means)
            i_min = np.around((alpha/2.0)*n_samples)
            i_max = np.around((1.0-alpha/2.0)*n_samples)
            self.intervals[interval, 0] = sorted_means[i_min]
            self.intervals[interval, 1] = sorted_means[i_max]

    def plot(self):
        pass
