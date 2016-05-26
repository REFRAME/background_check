import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve


def cross_entropy(p,q):
    return -np.sum(p * np.log(np.clip(q, 1e-16, 1.0)), axis=1)


def average_cross_entropy(p,q):
    return -np.mean(p * np.log(np.clip(q, 1e-16, 1.0)))


def composite_average_cross_entropy(p1,q1,p2,q2):
    return average_cross_entropy(p1,q1) + average_cross_entropy(p2,q2)


def compute_cace(c1_rs, c1_ts, c2_ts, y):
    q1 = np.vstack([c1_rs, c1_ts])
    p1 = np.vstack([np.ones((len(c1_rs),1)), np.zeros((len(c1_ts),1))])
    p1 = np.hstack([p1, 1-p1])

    q2 = c2_ts
    p2 = label_binarize(y, np.unique(y))
    ace1 = average_cross_entropy(p1,q1)
    ace2 = average_cross_entropy(p2,q2)
    return ace1, ace2, ace1+ace2


def calculate_mo_ce(clas_k, clas_u, confidence_k, confidence_u, labels_k):
    """ Multi-Output Cross-entropy
    Args:
        clas_k ([[float]]): Matrix with the predictions of the known data with
            one column per class and one row per sample.
        clas_u ([[float]]): Matrix with the predictions of the unknown data with
            one column per class and one row per sample.
        confidence_k ([float]): Vector with confidence values for known data
        confidence_u ([float]): Vector with confidence values for unknown data
        labels_k ([int]): True labels for the known data

    Returns:
        (float): Average cross entropy of a baseline model
        (float): Average cross entropy for the given predictions
    """
    classes = np.unique(labels_k)
    n_classes = len(classes)
    p_clas_k = label_binarize(labels_k, classes)
    p_clas_u = np.ones((clas_u.shape[0], p_clas_k.shape[1])) / n_classes
    p_clas = np.vstack((p_clas_k, p_clas_u))

    q_clas = np.vstack((clas_k, clas_u))
    ce_clas = cross_entropy(p_clas, q_clas)

    p_rej = np.vstack([np.zeros((np.alen(labels_k), 1)),
                       np.ones((np.alen(clas_u), 1))])
    p_rej = np.hstack([p_rej, 1-p_rej])

    q_rej = np.vstack((confidence_k, confidence_u))
    ce_rej = cross_entropy(p_rej, q_rej)

    q_bas_rej = np.zeros(q_rej.shape)
    q_bas_rej[:, 1] = 1.0
    ce_bas_rej = cross_entropy(p_rej, q_bas_rej)

    ce_bas = np.sum(ce_clas + ce_bas_rej) / (np.alen(ce_clas) * 2.0)
    ce_cco = np.sum(ce_clas + ce_rej) / (np.alen(ce_clas) * 2.0)

    return ce_bas, ce_cco, ce_clas.mean(), ce_rej.mean(), ce_bas_rej.mean()


def one_vs_rest_roc_curve(y,p, pos_label=0):
    """Returns the roc curve of class 0 vs the rest of the classes"""
    aux = np.zeros_like(y)
    aux[y!=pos_label] = 1
    return roc_curve(aux, p, pos_label=0)
