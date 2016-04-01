import numpy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def generate_gaussians(means=[[0,0]], covs=[[[1,0],[0,1]]], samples=[1],
                       holes=[0,0]):
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
    num_classes = len(samples)
    if sum(holes) > 0:
      if len(holes) != num_classes:
        raise Exception('The number of holes needs to be the same as the'
                        'number of classes')
    else:
      holes = numpy.zeros(num_classes)

    X = numpy.empty(shape=(sum(samples),numpy.shape(means)[1]))
    Y = numpy.empty(shape=(sum(samples)))
    cumsum = 0
    for i in range(num_classes):
        new_cumsum = cumsum + samples[i]
        Y[cumsum:new_cumsum] = i
        if holes[i] != 0:
          loop = True
          hole_origin = numpy.random.multivariate_normal(
                                    means[i], covs[i], size=(1))
          valid_samples = 0
          cumsum_partial = cumsum
          new_cumsum_partial = cumsum
          while new_cumsum_partial < new_cumsum:
            x_samples = numpy.random.multivariate_normal(
                                    means[i], covs[i], size=(samples[i]))
            x_samples = x_samples[numpy.sum((x_samples-hole_origin)**2,
              axis=1)**(1./2) > holes[i]]

            if (len(x_samples) + cumsum_partial) > new_cumsum:
              x_samples = x_samples[:new_cumsum - cumsum_partial]
            new_cumsum_partial = cumsum_partial + len(x_samples)
            X[cumsum_partial:new_cumsum_partial] = x_samples
            cumsum_partial = new_cumsum_partial
        else:
          X[cumsum:new_cumsum] = numpy.random.multivariate_normal(
                                   means[i], covs[i], size=(samples[i]))

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
