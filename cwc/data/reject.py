#!/usr/bin/python

import numpy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def hypersphere_distribution(size, dimensions):
    '''
    Reference:
    http://math.stackexchange.com/questions/87230/picking-random-points-in-the-volume-of-sphere-with-uniform-probability
    '''
    U = numpy.random.rand(size)
    X = numpy.random.normal(size=(dimensions, size))

    sphere1 = numpy.power(U, 1.0/dimensions)
    sphere2 = sphere1*X
    sphere3 = sphere2/numpy.sqrt(numpy.sum(X**2.0, axis=0))

    return sphere3

def hypersphere_boundary_distribution(size, dimensions):
    normal_deviates = numpy.random.normal(size=(dimensions,size))

    radius = numpy.sqrt((normal_deviates**2).sum(axis=0))
    points = normal_deviates/radius

def create_reject_data(X, proportion, method):
    num_reject = X.shape[0]*proportion
    dimensions = X.shape[1]
    if method == 'uniform_hcube':
        return numpy.random.rand(dimensions, num_reject)
    elif method == 'uniform_hsphere':
        return hypershpere_distribution(num_reject, dimensions)

def test_hypersphere():
    x = hypersphere_distribution(1000,3)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x[0,:], x[1,:], x[2,:])

    plt.show()

if __name__ == '__main__':
    test_hypersphere()



