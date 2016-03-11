import numpy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
plt.ion()

from sklearn.decomposition import PCA

from cwc.data import data
from cwc.data import reject


if __name__ == '__main__':
    samples = [1000, 500]
    x, y = data.generate_gaussian_data(means=[[3,3,3],[5,5,5]],
                                  covs=[[[2,1,0],[1,1,0],[0,0,1]],
                                        [[1,0,0],[0,1,1],[0,1,2]]],
                                  samples=samples)
    fig = plt.figure('data')
    ax = fig.add_subplot(111, projection='3d')

    for k, c in [(0, 'r'), (1, 'b')]:
        index = (y == k)
        ax.scatter(x[index,0], x[index,1], x[index,2], c=c)

    pca = PCA(n_components=2, whiten=False)
    pca.fit(x)
    print(pca.get_covariance())

    fig = plt.figure('pca')
    x_transform = pca.transform(x)
    for k, c in [(0, 'r'), (1, 'b')]:
        index = (y == k)
        plt.scatter(x_transform[index,0], x_transform[index,1], c=c)


    x_transform_means = x_transform.mean(axis=0)
    x_transform_std = x_transform.std(axis=0)

    r = reject.hypersphere_distribution(numpy.sum(samples),pca.n_components,
                                        radius=1.1)


    # FIXME : look at the details of generating the reject data
    r = x_transform_std*numpy.max(numpy.cov(x_transform.T))*r+x_transform_means

    plt.scatter(r[:,0], r[:,1], c='m')

    from IPython import embed
    embed() 
