from __future__ import division
import numpy as np

def precision(tp, tn, fp, fn):
    return tp/(tp + fp)

def recall(tp, tn, fp, fn):
    return tp/(tp + fn)

def precision_gain(tp, fn, fp, tn):
    """Calculates Precision Gain from the contingency table

    This function calculates Precision Gain from the entries of the contingency
    table: number of true positives (TP), false negatives (FN), false positives
    (FP), and true negatives (TN). More information on Precision-Recall-Gain
    curves and how to cite this work is available at
    http://www.cs.bris.ac.uk/~flach/PRGcurves/.
    """
    n_pos = tp + fn
    n_neg = fp + tn
    prec_gain = 1. - (n_pos * fp) / (n_neg * tp)
    if np.alen(prec_gain) > 1:
        prec_gain[tn + fn == 0] = 0
    elif tn + fn == 0:
        prec_gain = 0
    return prec_gain


def recall_gain(tp, fn, fp, tn):
    """Calculates Recall Gain from the contingency table

    This function calculates Recall Gain from the entries of the contingency
    table: number of true positives (TP), false negatives (FN), false positives
    (FP), and true negatives (TN). More information on Precision-Recall-Gain
    curves and how to cite this work is available at
    http://www.cs.bris.ac.uk/~flach/PRGcurves/.

    Args:
        tp (float) or ([float]): True Positives
        fn (float) or ([float]): False Negatives
        fp (float) or ([float]): False Positives
        tn (float) or ([float]): True Negatives
    Returns:
        (float) or ([float])
    """
    n_pos = tp + fn
    n_neg = fp + tn
    rg = 1. - (n_pos * fn) / (n_neg * tp)
    if np.alen(rg) > 1:
        rg[tn + fn == 0] = 1
    elif tn + fn == 0:
        rg = 1
    return rg


def sort_scores(pos_scores, neg_scores):
    to_sort = np.array([1.0-pos_scores, -neg_scores])
    new_order = np.argsort(to_sort)[0]
    return new_order


def create_segments(labels, pos_scores, neg_scores):
    n = np.alen(labels)
    # reorder labels and pos_scores by decreasing pos_scores, using increasing neg_scores in breaking ties
    new_order = sort_scores(pos_scores, neg_scores)
    labels = labels[new_order]
    pos_scores = pos_scores[new_order]
    neg_scores = neg_scores[new_order]
    # create a table of segments
    segments = {'pos_score': np.zeros(n), 'neg_score': np.zeros(n), 'pos_count': np.zeros(n), 'neg_count': np.zeros(n)}
    j = -1
    for i, label in enumerate(labels):
        if (i == 0) or (pos_scores[i - 1] != pos_scores[i]) or (neg_scores[i-1] != neg_scores[i]):
            j += 1
            segments['pos_score'][j] = pos_scores[i]
            segments['neg_score'][j] = neg_scores[i]
        if label == 0:
            segments['neg_count'][j] += 1
        else:
            segments['pos_count'][j] += 1
    segments['pos_score'] = segments['pos_score'][0:j+1]
    segments['neg_score'] = segments['neg_score'][0:j+1]
    segments['pos_count'] = segments['pos_count'][0:j+1]
    segments['neg_count'] = segments['neg_count'][0:j+1]
    return segments


def get_point(points, index):
    keys = points.keys()
    point = np.zeros(np.alen(keys))
    key_indices = dict()
    for i, key in enumerate(keys):
        point[i] = points[key][index]
        key_indices[key] = i
    return [point, key_indices]


def insert_point(new_point, key_indices, points, precision_gain=0, recall_gain=0):
    for key in key_indices.keys():
        points[key] = np.insert(points[key], 0, new_point[key_indices[key]])
    points['precision_gain'][0] = precision_gain
    points['recall_gain'][0] = recall_gain
    points['is_crossing'][0] = 1
    new_order = sort_scores(points['pos_score'], points['neg_score'])
    for key in points.keys():
        points[key] = points[key][new_order]
    return points


def create_crossing_points(points, n_pos, n_neg):
    n = n_pos+n_neg
    points['is_crossing'] = np.zeros(np.alen(points['pos_score']))
    # introduce a crossing point at the crossing through the y-axis
    j = np.amin(np.where(points['recall_gain'] >= 0)[0])
    if points['recall_gain'][j] > 0:  # otherwise there is a point on the boundary and no need for a crossing point
        [point_1, key_indices_1] = get_point(points, j)
        [point_2, key_indices_2] = get_point(points, j-1)
        delta = point_1 - point_2
        if delta[key_indices_1['TP']] > 0:
            alpha = (n_pos * n_pos / n-points['TP'][j-1]) / delta[key_indices_1['TP']]
        else:
            alpha = 0.5

        new_point = point_2 + alpha*delta

        new_prec_gain = precision_gain(new_point[key_indices_1['TP']], new_point[key_indices_1['FN']],
                                       new_point[key_indices_1['FP']], new_point[key_indices_1['TN']])
        points = insert_point(new_point, key_indices_1, points, precision_gain=new_prec_gain)

    # now introduce crossing points at the crossings through the non-negative part of the x-axis
    x = points['recall_gain']
    y = points['precision_gain']
    temp_y_0 = np.append(y, 0)
    temp_0_y = np.append(0, y)
    temp_1_x = np.append(1, x)
    for i in np.where(np.logical_and((temp_y_0 * temp_0_y < 0), (temp_1_x >= 0)))[0]:
        cross_x = x[i-1] + (-y[i-1]) / (y[i] - y[i-1]) * (x[i] - x[i-1])
        [point_1, key_indices_1] = get_point(points, i)
        [point_2, key_indices_2] = get_point(points, i-1)
        delta = point_1 - point_2
        if delta[key_indices_1['TP']] > 0:
            alpha = (n_pos * n_pos / (n - n_neg * cross_x) - points['TP'][i-1]) / delta[key_indices_1['TP']]
        else:
            alpha = (n_neg / n_pos * points['TP'][i-1] - points['FP'][i-1]) / delta[key_indices_1['FP']]

        new_point = point_2 + alpha*delta
        new_rec_gain = recall_gain(new_point[key_indices_1['TP']], new_point[key_indices_1['FN']],
                                   new_point[key_indices_1['FP']], new_point[key_indices_1['TN']])
        points = insert_point(new_point, key_indices_1, points, recall_gain=new_rec_gain)
    return points


def calculate_prg_points(labels, pos_scores, neg_scores=[]):
    if np.alen(neg_scores) == 0:
        neg_scores = -pos_scores
    n = np.alen(labels)
    n_pos = np.sum(labels)
    n_neg = n - n_pos
    # convert negative labels into 0s
    labels = 1 * (labels == 1)
    segments = create_segments(labels, pos_scores, neg_scores)
    # calculate recall gains and precision gains for all thresholds
    points = dict()
    points['pos_score'] = np.insert(segments['pos_score'], 0, np.inf)
    points['neg_score'] = np.insert(segments['neg_score'], 0, -np.inf)
    points['TP'] = np.insert(np.cumsum(segments['pos_count']), 0, 0)
    points['FP'] = np.insert(np.cumsum(segments['neg_count']), 0, 0)
    points['FN'] = n_pos - points['TP']
    points['TN'] = n_neg - points['FP']
    points['precision_gain'] = precision_gain(points['TP'], points['FN'], points['FP'], points['TN'])
    points['recall_gain'] = recall_gain(points['TP'], points['FN'], points['FP'], points['TN'])
    points = create_crossing_points(points, n_pos, n_neg)
    return points
