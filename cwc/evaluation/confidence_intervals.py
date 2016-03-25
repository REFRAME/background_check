from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import warnings


class ConfidenceIntervals:
    """This class represents non-parametric confidence intervals built
     for the means of the values as parameter.

    Args:
        values ([[float]]): Matrix (n x p), where "p" is the number of
         samples and "n" is the number of values in each sample.
        x_labels ([str]): Names of the samples. Length must be the same
         as the number of columns in values.
        n_samples (int): Number of samples used to be build the intervals.
        alpha (float): Significance level.

    Attributes:
        intervals ([[float]]): Matrix (p x 2), where "p" is the number of
         intervals and the columns contain the minimum and maximum values of
         the intervals, respectively.
        labels ([str]): Names of the intervals.

    """
    def __init__(self, values, x_labels, n_samples=100, alpha=0.05):
        n_intervals = np.alen(x_labels)
        sample_size = np.alen(values)
        self.intervals = np.zeros((n_intervals, 2))
        self.labels = x_labels
        for interval in np.arange(n_intervals):
            v = values[:, interval]
            means = np.zeros(n_samples)
    
            for i in np.arange(n_samples):
                sample = np.random.choice(v, sample_size)
                means[i] = np.mean(sample)
            
            sorted_means = np.sort(means)
            i_min = np.around((alpha/2.0)*n_samples)
            i_max = np.around((1.0-alpha/2.0)*n_samples)
            if i_max == n_samples:
                i_max -= 1
            self.intervals[interval, 0] = sorted_means[i_min]
            self.intervals[interval, 1] = sorted_means[i_max]

    @property
    def mins(self):
        """This property returns the minimum values of the intervals.

        Returns:
            [float]: The minimum values of the intervals.

        """
        return self.intervals[:, 0]

    @property
    def maxs(self):
        """This property returns the maximum values of the intervals.

        Returns:
            [float]: The maximum values of the intervals.

        """
        return self.intervals[:, 1]

    def plot(self, fig=None):
        """This method plots the confidence intervals, with the names
        on the x-axis.

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
        means = np.mean(self.intervals, 1)
        yerr = self.maxs - means
        ax = fig.add_subplot(111)
        n_intervals = np.alen(self.maxs)
        x = np.arange(n_intervals)
        ax.errorbar(x, means, yerr=yerr, fmt='o')
        plt.xticks(x, self.labels, rotation='horizontal')
        ax.set_xlim(x + [-1, 1])
        plt.show()
