import numpy as np
import matplotlib.pyplot as plt
from ..evaluation.metrics import one_vs_rest_roc_curve
from roc_analysis import plot_roc_curve

def plot_intersections(ax, intersections):
    ax.scatter(intersections[:,0], intersections[:,1])

def get_slope(line):
    return (line[3]-line[1])/(line[2]-line[0])

def get_slopes(lines):
    return [get_slope(line) for line in lines]

def find_first_segment(lines):
    left_min = lines[0]
    index = 0
    for i, line in enumerate(lines):
        if all(line[[0,1]] == left_min[[0,1]]):
            slope_min = get_slope(left_min)
            slope_2 = get_slope(line)
            if slope_2 < slope_min:
                left_min = line
                index = i
        elif all(line[[0,1]] <= left_min[[0,1]]):
            left_min = line
            index = i
    return index
#
# line segment intersection using vectors
# see Computer Graphics by F.S. Hill
#
from numpy import *
def perp( a ) :
    b = empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
def seg_intersect(a1,a2, b1,b2) :
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = dot( dap, db)
    num = dot( dap, dp )
    return (num / denom.astype(float))*db + b1

def find_all_intersections(line1, lines):
    intersections = np.empty((lines.shape[0], 2))
    for i, line in enumerate(lines):
        intersections[i] = seg_intersect(line1[[0,1]], line1[[2,3]],
                                         line[[0,1]], line[[2,3]])
    return intersections

def find_smallest_point_intersection(line1, lines, right_of=0):
    intersections = find_all_intersections(line1, lines)
    left_min = line1
    index = 0
    for i, line in enumerate(lines):
        if right_of <= intersections[i,0]:
            slope_min = get_slope(left_min)
            slope_2 = get_slope(line)
            if slope_2 < slope_min:
                left_min = line
                index = i

    return intersections[index], left_min


def plot_lower_envelope(lines, ax, show_segments=False):
# First segment of the line:
    lower_envelope = []
    lower_envelope.append([0,0])
    index = find_first_segment(lines)
    current_line = lines[index]
    if show_segments:
        ax.plot(current_line[[0,2]].T, current_line[[1,3]].T, 'k--')
    left_point = current_line[[0,1]]

    intersections = find_all_intersections(current_line, lines)
    if show_segments:
        plot_intersections(ax, intersections)

    indices = np.argsort(intersections, axis=0)[:,0]
    next_index = indices[0]
    next_line = lines[next_index]
    intersection = intersections[next_index]
    if show_segments:
        ax.plot(next_line[[0,2]].T, next_line[[1,3]].T, 'k--')

    lower_envelope.append(intersection)
    while next_line[3] != 0:
        current_line = next_line
        next_intersection = intersections[next_index]
        left_point = next_intersection[0]

        intersections = find_all_intersections(current_line, lines)
        if show_segments:
            plot_intersections(ax, intersections)
        indices = np.argsort(intersections, axis=0)[:,0]
        found = False
        i = 0
        while not found:
            next_intersection = intersections[indices[i],:]
            if next_intersection[0] > left_point:
                found = True
            else:
                i+=1
        indices = np.where(np.logical_and(intersections[:,0] == next_intersection[0], intersections[:,1] == next_intersection[1]))
        next_index = indices[0].max()
        next_line = lines[next_index]
        if show_segments:
            ax.plot(next_line[[0,2]].T, next_line[[1,3]].T, 'k--')
        next_intersection = intersections[next_index]
        lower_envelope.append(next_intersection)

    lower_envelope.append([1,0])
    lower_envelope = np.array(lower_envelope)
    ax.plot(lower_envelope[:,0], lower_envelope[:,1], 'ro-')
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])

    return lower_envelope

def skew_lines(fpr, tpr):
    Q_min = fpr
    Q_max = 1-tpr
    lines = np.vstack([np.vstack((np.zeros_like(Q_min), Q_min)),
                       np.vstack((np.ones_like(Q_max), Q_max))]).T
    return lines

def plot_skew_lines(y, scores, pos_label=0, lower_envelope=False, fig=None,
                    title=None):
    if fig is None:
        fig = plt.figure('skew_lines')
    fig.clf()

    roc = one_vs_rest_roc_curve(y,scores,pos_label)
    ax = fig.add_subplot(111)
    lines = skew_lines(roc[0], roc[1])
    ax.plot(lines[:,[0,2]].T, lines[:,[1,3]].T, '--', c='0.6')
    ax.set_xlabel('skew')
    ax.set_ylabel('$Q_{skew}$')
    if title is not None:
        ax.set_title(title)
    if lower_envelope:
        plot_lower_envelope(lines, ax)

    return fig
