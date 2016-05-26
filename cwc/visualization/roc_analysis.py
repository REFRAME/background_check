import matplotlib.pyplot as plt
from ..evaluation.metrics import one_vs_rest_roc_curve
from sklearn.metrics import auc

def convex_hull(points):
    """Computes the convex hull of a set of 2D points.
    Input: an iterable sequence of (x, y) pairs representing the points.
    Output: a list of vertices of the convex hull in counter-clockwise order,
      starting from the vertex with the lexicographically smallest coordinates.
    Implements Andrew's monotone chain algorithm. O(n log n) complexity.
    Source code from:
    https://en.wikibooks.org/wiki/Algorithm_Implementation/Geometry/Convex_hull/Monotone_chain
    """

    # Sort the points lexicographically (tuples are compared lexicographically).
    # Remove duplicates to detect the case we have just one unique point.
    points = sorted(set(points))

    # Boring case: no points or a single point, possibly repeated multiple times.
    if len(points) <= 1:
        return points

    # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
    # Returns a positive value, if OAB makes a counter-clockwise turn,
    # negative for clockwise turn, and zero if the points are collinear.
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return upper

def plot_roc_curve(y,p,fig=None,title='', pos_label=0):
    if fig is None:
        fig = plt.figure('roc_curve')
    fig.clf()

    roc = one_vs_rest_roc_curve(y, p, pos_label=pos_label)
    auroc = auc(roc[0], roc[1])
    ax = fig.add_subplot(111)
    ax.plot(roc[0], roc[1])
    ax.plot([0,1],[0,1], 'g--')
    upper_hull = convex_hull(zip(roc[0],roc[1]))
    rg_hull, pg_hull = zip(*upper_hull)
    plt.plot(rg_hull, pg_hull, 'r--')
    ax.set_title('{0} {1:.3f}'.format(title, auroc))
    ax.set_ylim([0, 1.01])
    ax.set_xlim([-0.01, 1.01])
    ax.grid(True)
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')

    return auroc

