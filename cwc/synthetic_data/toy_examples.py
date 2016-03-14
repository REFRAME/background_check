import numpy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def generate_gaussians(means=[[0,0]], covs=[[[1,0],[0,1]]], samples=[1]):
    """This function generates N Gaussian distributions with the specified
        mean, covariance and number of samples per Gaussian.

    Args:
        means ([[float]]): The D-dimensional means for each Gaussian.
        covs ([[[float]]]): The D-dimensional covariance matrix for each
            Gaussian.
        samples ([int]): The number of samples for each Gaussian.

    Returns:
        X ([[float]]): A matrix of size sum(samples) times D.
        Y ([int]): A vector of size sum(samples) indicating the Gaussian
            distribution where every X belongs.
    """
    X = numpy.empty(shape=(sum(samples),numpy.shape(means)[1]))
    Y = numpy.empty(shape=(sum(samples)))
    cumsum = 0
    for i in range(len(means)):
        new_cumsum = cumsum + samples[i]
        X[cumsum:new_cumsum] = numpy.random.multivariate_normal(
                                means[i], covs[i], size=(samples[i]))
        Y[cumsum:new_cumsum] = i
        cumsum = new_cumsum

    rand_indices = numpy.random.permutation(range(cumsum))
    X = X[rand_indices,:]
    Y = Y[rand_indices]

    return X, Y

def test_generate_gaussians():
    """This function generates two Gaussian distributions and plots them.
    """
    x, y = generate_gaussians(means=[[0,0,0],[1,1,1]],
                                  covs=[[[2,1,0],[1,1,0],[0,0,1]],
                                        [[1,0,0],[0,1,1],[0,1,2]]],
                                  samples=[1000, 500])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for k, c in [(0, 'r'), (1, 'b')]:
        index = (y == k)
        ax.scatter(x[index,0], x[index,1], x[index,2], c=c)

    plt.show()

if __name__ == '__main__':
    test_generate_gaussians()
