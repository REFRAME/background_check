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


def create_reject_data(X, proportion, method, pca=False, pca_components=0,
                       pca_variance=0, hshape_cov=0, hshape_prop_in=0,
                       hshape_multiplier=1):
    """Generates random samples with the specified distribution.

    This function generates data points with the specified distribution from
    some available options.

    Args:
        X ([[int]]): Matrix with the original data-points, with dimensions .
        proportion (float): Proportion of reject data to generate.
        method (string):
            - 'uniform_hcube': Uniform hypercube
            - 'uniform_hsphere': Uniform hypersphere
        PCA (bool): Indicates if the reject points should be generated in a
            lower dimensionality space.
        pca_components (int): e.g. 2 Number of dimensions of the hyper-shape in
            the lower dimensionality space (this option and pca_variance are
            mutually exclusive if both options are set at the same time the
            function will raise an Exception).
        pca_variance (float): e.g. 0.99 Proportion of variance (0.0,1.0) to
            keep in the lower dimensionality space (this option and pca_
            components are mutually exclusive if both options are set at the
            same time the function will raise an Exception).
        hshape_cov (float): e.g. 1.1
        hshape_prop_in (float): Proportion of data points that need to be
            inside of the hyper-shape.
        hshape_multiplier (float): Increases or decreases the distance to the
            boundary of the hyper-shape.

    Returns:
        ([[float]]): A matrix of size (size x dimensions)
    """
    if pca == True:
        if pca_components and pca_variance:
            raise Exception('Options pca_components and pca_variance are'
                           'mutually exclusive')
        elif pca_components:
            dimensions = pca_components
            pc = PCA(n_components=dimensions, whiten=True)
            pc.fit(X)
        elif pca_variance:
            dimensions = X.shape[1]
            pc = PCA(n_components=dimensions, whiten=True)
            pc.fit(X)
            dimensions = numpy.argmax(pc.explained_variance_ratio_.cumsum() >=
                                      pca_variance)+1
        else:
            raise Exception('If PCA is selected pca_components or pca_variance '
                           'need to be specified')
    else:
        dimensions = X.shape[1]

    num_reject = X.shape[0]*proportion

    if hshape_prop_in and hshape_cov:
        raise Exception('Options hshape_prop_in and hshape_cov are'
                       'mutually exclusive')

    if hshape_prop_in:
        if pca == True:
            X_transform = pc.transform(X)
            distances = numpy.linalg.norm(X_transform, axis=1)
        else:
            distances = numpy.linalg.norm(X, axis=1)
        distances.sort()
        radius = distances[int(X.shape[0]*hshape_prop_in)-1]
    else:
        radius = 1

    radius *= hshape_multiplier

    if method == 'uniform_hcube':
        r = hypercube_distribution(num_reject, dimensions, a=-radius/2,
                b=radius/2)

        if hshape_cov:
            # FIXME check the equation of the variance for hypercube
            r *= numpy.sqrt(hshape_cov/(numpy.mean(r**2)))
    elif method == 'uniform_hsphere':
        r = hypersphere_distribution(num_reject, dimensions,
                                            radius=radius)
        # TODO compute the value of r before creating the hypersphere
        if hshape_cov:
            r *= numpy.sqrt(hshape_cov/(numpy.mean(r**2)))
    else:
       raise Exception("Method to generate reject data unknown "
                       "(method=\'{}\')".format(method))

    if pca == True:
        if pca_variance and dimensions < X.shape[0]:
            redundant_dim = numpy.zeros((num_reject, X.shape[1] - dimensions))
            r = numpy.hstack((r,redundant_dim))

        r = pc.inverse_transform(r)

    return r

def test_hypersphere():
    x = hypersphere_distribution(1000,3)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x[:,0], x[:,1], x[:,2])

    plt.show()

if __name__ == '__main__':
    test_hypersphere()
