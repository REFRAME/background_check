import numpy as np
import cwc.evaluation.rgrpg as rgrpg


if __name__ == "__main__":
    predicted_labels_python = np.genfromtxt("datasets/predicted_labels_python.csv", delimiter=",", skip_header=True)
    step1_labels = predicted_labels_python[:, 4]
    step1_scores = predicted_labels_python[:, 0]
    step2_labels = predicted_labels_python[:, 5][step1_labels == 1]
    step2_scores = predicted_labels_python[:, 2][step1_labels == 1]
    [recall_gains, precision_gains, rocs, areas] = rgrpg.build_rgrpg_surface(step1_labels, step1_scores, step2_labels, step2_scores)
    # print rgrpg.calculate_volume(recall_gains, precision_gains, areas)
    # rgrpg.plot_rgrpg_2d(recall_gains, precision_gains, areas)
    rgrpg.plot_rgrpg_3d(recall_gains, precision_gains, rocs)
