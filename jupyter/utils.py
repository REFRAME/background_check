import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import norm

def draw_in_row(fruits, sizes):
    indices = np.argsort(sizes)
    sizes_sorted = sizes[indices]
    fruits_sorted = fruits[indices]

    images = map(Image.open, ['./images/'+fruit+'.jpg' for fruit in
                              fruits_sorted])
    widths, heights = zip(*(i.size for i in images))

    unit_width = 200

    drawing_sizes = np.array(sizes_sorted*unit_width, dtype=int)
    total_width = drawing_sizes.sum()+unit_width
    max_height = drawing_sizes.max()

    new_im = Image.new('RGB', (total_width, max_height),color=(255,255,255,0))

    drawn_ball = False
    x_offset = 0
    for im, drawing_size in zip(images, drawing_sizes):
        if not drawn_ball and drawing_size >= unit_width:
            im_aux = Image.open('./images/tennis_ball.jpg')
            im_aux = im_aux.resize([unit_width, unit_width])
            new_im.paste(im_aux, (x_offset,0))
            drawn_ball = True
            x_offset += im_aux.size[0]
        im = im.resize([drawing_size,drawing_size])
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]

    ax = plt.figure(figsize=(7,5), dpi=80)
    plt.imshow(new_im)
    plt.axis('off')


class NormalDistribution(object):
    def __init__(self, x=None, mu=None, sigma=None):
        if x is not None:
            self.fit(x)
        else:
            if mu is not None:
                self.mu = mu
            if sigma is not None:
                self.sigma = sigma

    def fit(self, x):
        self.mu = x.mean()
        self.sigma = x.std()

    def pdf(self,x):
        return norm.pdf(x, loc=self.mu, scale=self.sigma)

    def sample(self, n):
        return norm.rvs(loc=self.mu, scale=self.sigma, size=n)

class MixtureGaussians(object):
    def __init__(self, gaussians, priors=None):
        self.gaussians = gaussians
        if priors is None:
            self.priors = np.ones(self.n_gaussians)/self.n_gaussians
        else:
            self.priors = priors

    def add_gaussian(self, gaussian, prior=None):
        self.gaussians.append(gaussian)
        if prior is None:
            self.priors = np.ones(self.n_gaussians)/self.n_gaussians
        else:
            self.priors.append(prior)

    @property
    def n_gaussians(self):
        return len(self.gaussians)

    @property
    def priors_norm(self):
        return self.priors/np.sum(self.priors)

    def pdf(self,x):
        result = np.zeros_like(x, dtype=float)
        for prior, gaussian in zip(self.priors_norm, self.gaussians):
            result += gaussian.pdf(x)*prior
        return result

    def sample(self, n):
        result = np.zeros(n, dtype=float)
        ns = np.random.multinomial(n, self.priors_norm)
        index = 0
        for n_i, prior, gaussian in zip(ns, self.priors_norm, self.gaussians):
            result[index:index+n_i] = gaussian.sample(n_i)
            index += n_i
        return result



def plot_confusion_matrix(cm, labels, title='Confusion matrix',
        cmap=plt.cm.Blues, show_accuracy=True):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    res = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.set_aspect(1)

    cb = fig.colorbar(res)
    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels)

    fig.tight_layout()

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    if show_accuracy:
        ax.set_title("{} (acc={:2.2f}%)".format(title,
            np.true_divide(100*np.diag(cm).sum(),cm.sum())))
    else:
        ax.set_title(title)

    width, height = cm.shape

    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(cm[x][y]), xy=(y, x),
                            horizontalalignment='center',
                            verticalalignment='center')


if __name__ == '__main__':
    pass
