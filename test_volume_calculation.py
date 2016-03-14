import numpy as np
import cwc.evaluation.volume as vol


if __name__ == "__main__":
    predicted_labels_python = np.genfromtxt("datasets/predicted_labels_python.csv", delimiter=",", skip_header=True)
    step1_labels = predicted_labels_python[:, 4]
    step1_scores = predicted_labels_python[:, 0]
    step2_labels = predicted_labels_python[:, 5]
    step2_scores = predicted_labels_python[:, 2]
    print vol.calculate_volume(step1_labels, step1_scores, step2_labels, step2_scores)
    # points = calculate_prg_points(predicted_labels_python[:, 4], predicted_labels_python[:, 0])
    # n = np.alen(points['recall_gain'])
    # print 'pos_score, neg_score, TP, FP, FN, TN, precision_gain, recall_gain'
    # for i in np.arange(n):
    #     print str(points['pos_score'][i]) + ", " + str(points['neg_score'][i]) + ", " \
    #           + str(points['TP'][i]) + ", " + str(points['FP'][i]) + ", " \
    #           + str(points['FN'][i]) + ", " + str(points['TN'][i]) + ", " \
    #           + str(points['precision_gain'][i]) + ", " + str(points['recall_gain'][i])
