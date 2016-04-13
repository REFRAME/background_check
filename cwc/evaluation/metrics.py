import numpy as np
from sklearn.preprocessing import label_binarize

def average_cross_entropy(p,q):
    return - np.mean(p*np.log(np.clip(q, 1e-16, 1.0)))


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


