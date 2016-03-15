import numpy as np
from cwc.evaluation.rgrpg import RGRPG


if __name__ == "__main__":
    predicted_labels_python = np.genfromtxt("datasets/predicted_labels_python.csv", delimiter=",", skip_header=True)
    step1_labels = predicted_labels_python[:, 4]
    step1_scores = predicted_labels_python[:, 0]
    step2_labels = predicted_labels_python[:, 5][step1_labels == 1]
    step2_scores = predicted_labels_python[:, 2][step1_labels == 1]
    rgrpg = RGRPG(step1_labels, step1_scores, step2_labels, step2_scores)
    print rgrpg.calculate_volume()
    rgrpg.plot_rgrpg_2d()
    rgrpg.plot_rgrpg_3d()
