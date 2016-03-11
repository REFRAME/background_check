import numpy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def generate_gaussian_data(means=[[0,0]], covs=[[[1,0],[0,1]]], samples=[1]):
    print('means shape = {} len = {}'.format(numpy.shape(means), len(means)))
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

def test_generate_gaussian_data():
    x, y = generate_gaussian_data(means=[[0,0,0],[1,1,1]],
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
    test_generate_gaussian_data()
