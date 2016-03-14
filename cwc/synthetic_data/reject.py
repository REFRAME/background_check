import numpy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def hypercube_distribution(size, dimensions, a=-0.5, b=0.5):
    """Generates random samples from a hypercube.

    This function generates uniformly distributed data inside an hypercube of
    the specified number of dimensions.

    Args:
        size (int): The number of data samples.
        dimensions (int): The number of dimensions of the hypercube.
        a (float): Minimum value of the hypercube
        b (float): Maximum value of the hypercube

    Returns:
        ([[float]]): A matrix of size (size x dimensions)
    """
    return numpy.random.uniform(a,b,size=(size, dimensions))


def hypersphere_distribution(size, dimensions, radius=1.0):
    """Generates random samples from a hypersphere.

    This function generates uniformly distributed data-points inside a
    hypersphere of the specified number of dimensions and radius centered on
    zero.

    Args:
        size (int): The number of data-points.
        dimensions (int): The number of dimensions of the hypersphere.
        radius (float): Radius of the hypersphere

    Returns:
        ([[float]]): A matrix of size (size x dimensions)

    Reference:
        http://math.stackexchange.com/questions/87230/
        picking-random-points-in-the-volume-of-sphere-with-uniform-probability
    """
    U = numpy.random.rand(size)
    X = numpy.random.normal(size=(size, dimensions))

    sphere = radius*numpy.power(U, 1.0/dimensions).reshape((-1,1))
    sphere = sphere*X
    sphere = sphere/numpy.sqrt(numpy.sum(X**2.0, axis=1)).reshape((-1,1))

    return sphere


def hypersphere_boundary_distribution(size, dimensions):
    """Generates random samples from the boundary of a hypersphere.

    This function generates uniformly distributed data-points on the perimeter
    of a hypersphere of the specified number of dimensions radius 1 and
    centered on the origin.

    Args:
        size (int): The number of data-points.
        dimensions (int): The number of dimensions of the hypersphere.

    Returns:
        ([[float]]): A matrix of size (size x dimensions)
    """
    normal_deviates = numpy.random.normal(size=(dimensions,size))

    radius = numpy.sqrt((normal_deviates**2).sum(axis=0))
    points = normal_deviates/radius
    return points


def create_reject_data(X, proportion, method):
    """Generates random samples with the specified distribution.

    This function generates data points with the specified distribution from
    some available options.

    Args:
        X ([[int]]): Matrix with the original data-points, with dimensions .
        proportion (float): Proportion of reject data to generate.
        method (string):
            - 'uniform_hcube': Uniform hypercube
            - 'uniform_hsphere': Uniform hypersphere

    Returns:
        ([[float]]): A matrix of size (size x dimensions)
    """
    num_reject = X.shape[0]*proportion
    dimensions = X.shape[1]
    if method == 'uniform_hcube':
        return hypercube_distribution(num_reject, dimensions)
    elif method == 'uniform_hsphere':
        pca = PCA(n_components=2, whiten=True)
        pca.fit(X)

        X_transform = pca.transform(X)

        radius=1
        r = hypersphere_distribution(num_reject, pca.n_components,
                                            radius=radius)
        # TODO compute the value of r before creating the hypersphere
        r *= numpy.sqrt(1.1/(numpy.mean(r**2)))

        return pca.inverse_transform(r)

def test_hypersphere():
    x = hypersphere_distribution(1000,3)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x[:,0], x[:,1], x[:,2])

    plt.show()

if __name__ == '__main__':
    test_hypersphere()
