import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def generate_gaussians(means=[[0,0]], covs=[[[1,0],[0,1]]], samples=[1],
                       holes=[0,0], hole_centers=None, shuffle=True):
    """This function generates N Gaussian distributions with the specified
        mean, covariance and number of samples per Gaussian.

    Args:
        means ([[float]]): The D-dimensional means for each Gaussian.
        covs ([[[float]]]): The D-dimensional covariance matrix for each
            Gaussian.
        samples ([int]): The number of samples for each Gaussian.
        holes ([float]): Specify the radius of a hole per each Gaussian
        hole_centers ([[float]]): Specifies the center of each hole, if this is
            not specified the center is selected randomly.

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
        holes = np.zeros(num_classes)

    X = np.empty(shape=(sum(samples),np.shape(means)[1]))
    Y = np.empty(shape=(sum(samples)), dtype=int)
    cumsum = 0
    if hole_centers == None:
        hole_centers = np.empty_like(means)
        generate_centers = True
    else:
        generate_centers = False
    for i in range(num_classes):
        new_cumsum = cumsum + samples[i]
        Y[cumsum:new_cumsum] = i
        if holes[i] != 0:
          loop = True
          if generate_centers:
              hole_centers[i] = np.random.multivariate_normal(
                                    means[i], covs[i], size=(1))
          valid_samples = 0
          cumsum_partial = cumsum
          new_cumsum_partial = cumsum
          while new_cumsum_partial < new_cumsum:
            x_samples = np.random.multivariate_normal(
                                    means[i], covs[i], size=(samples[i]))
            x_samples = x_samples[np.sum((x_samples-hole_centers[i])**2,
              axis=1)**(1./2) > holes[i]]

            if (len(x_samples) + cumsum_partial) > new_cumsum:
                x_samples = x_samples[:new_cumsum - cumsum_partial]
            new_cumsum_partial = cumsum_partial + len(x_samples)
            X[cumsum_partial:new_cumsum_partial] = x_samples
            cumsum_partial = new_cumsum_partial
        else:
            X[cumsum:new_cumsum] = np.random.multivariate_normal(
                                   means[i], covs[i], size=(samples[i]))

        cumsum = new_cumsum

    if shuffle:
        rand_indices = np.random.permutation(range(cumsum))
        X = X[rand_indices,:]
        Y = Y[rand_indices]

    return X, Y, hole_centers

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


def generate_example(example=1, hole_centers=None):
    if example == 1:
        holes=[2,1]
        samples = [1800,         # Class 1
                   1000]         # Class 2
        means = [[2,2,2],       # Class 1
                 [2,2,3]]       # Class 2
        covs = [[[1,0,0],       # Class 1
                 [0,2,1],
                 [0,1,1]],
                [
                 [1,1,0],       # Class 2
                 [1,2,0],
                 [0,0,1]]]
    elif example == 2:
        holes=[2,2,1]
        samples = [400,         # Class 1
                   400,         # Class 2
                   400]         # Class 3
        means = [[0,0],       # Class 1
                 [-2,0],       # Class 2
                 [3,4]]       # Class 3
        covs = [[[3,-1],       # Class 1
                 [-1,2]],
                [[3,0],       # Class 2
                 [0,3]],
                [[2,1],       # Class 3
                 [1,2]]]
    elif example == 3:
        holes=[2,1]
        samples = [1800,         # Class 1
                   100]         # Class 2
        means = [[0,0],       # Class 1
                 [7,4]]       # Class 2
        covs = [[[3,-1],       # Class 1
                 [-1,2]],
                [[2,1],       # Class 2
                 [1,2]]]
    elif example == 4:
        holes=[4,5]
        samples = [1800,         # Class 1
                   1600]         # Class 2
        means = [[0,0,0,0,0,0,0,0,0,0],       # Class 1
                 [2,2,2,2,2,2,2,2,2,2]]       # Class 2
        A1 = np.random.rand(10,10)
        A2 = np.random.rand(10,10)
        covs = [np.dot(A1,A1.transpose()),
               np.dot(A2,A2.transpose())]
    elif example == 5:
        holes=[2,2,1]
        samples = [200,         # Class 1
                   200,         # Class 2
                   200]         # Class 3
        means = [[0,0],       # Class 1
                 [-3,-3],       # Class 2
                 [2,0]]       # Class 3
        covs = [[[3,-1],       # Class 1
                 [-1,2]],
                [[3,0],       # Class 2
                 [0,3]],
                [[2,1],       # Class 3
                 [1,2]]]

    elif example == 6:
        holes=[3,0]
        samples = [2000,         # Class 1
                   2000]         # Class 2
        means = [[0,0],       # Class 1
                 [0,0]]       # Class 2
        covs = [[[4,0],       # Class 1
                 [0,4]],
                [[2,0],       # Class 2
                 [0,2]]]
    elif example == 7:
        holes=[0,0]
        samples = [2000,         # Class 1
                   2000]         # Class 2
        means = [[0,0],       # Class 1
                 [1,1]]       # Class 2
        covs = [[[1,0],       # Class 1
                 [0,1]],
                [[1,0],       # Class 2
                 [0,1]]]
    elif example == 8:
        holes=[3,2,1,2]
        samples = [800,         # Class 1
                   600,         # Class 2
                   600,         # Class 3
                   300]         # Class 4
        means = [[-3,0],       # Class 1
                 [2,2],       # Class 2
                 [-3,0],       # Class 3
                 [3,-4]]       # Class 4
        covs = [[[3,0.3],       # Class 1
                 [0.3,2]],
                [[3,0],       # Class 2
                 [0,3]],
                [[1,-0.2],       # Class 3
                 [-0.2,1]],
                [[2,-0.5],       # Class 4
                 [-0.5,2]]]
    elif example == 9:
        classes = 5
        dimensions = 10
        holes = np.random.rand(classes)*2
        samples = np.random.randint(1000,1001,classes)
        means = np.random.rand(classes, dimensions)*2
        covs = (np.random.rand(classes, dimensions, dimensions)-0.5)*2
        for i, cov in enumerate(covs):
            covs[i] = np.dot(cov, cov.transpose())
    else:
        raise Exception('Example {} does not exist'.format(example))

    x, y, hole_centers = generate_gaussians(
                          means=means, covs=covs, samples=samples, holes=holes,
                            hole_centers=hole_centers)
    if example==5:
        y[y==2] = 1
    return x, y, hole_centers

if __name__ == '__main__':
    test_generate_gaussians()
