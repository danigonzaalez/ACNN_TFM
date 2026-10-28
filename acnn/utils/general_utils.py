import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        return super(NumpyEncoder, self).default(obj)


def calculate_seg_mean_dev(seg_results):
    # Initialize a dictionary to store lists of values for each key
    results_by_key = {key: [] for key in seg_results[0].keys()}

    # Collect values for each key across all experiments
    for result in seg_results:
        for key, value in result.items():
            results_by_key[key].append(value)

    # Calculate mean and standard deviation for each key
    stats = {}
    for key, values in results_by_key.items():
        stats[key] = {"mean": np.mean(values), "std_dev": np.std(values)}

    return stats

# COMMENTED: ONLY FOR ANOMALY DETECTION
# def ad_accuracies(truth, output):
#     # Anomaly detection metrics
#     ad_results_dict = {}

#     # Calculate TP, TN, FP, FN
#     # print count of unique values in output and truth
#     print(f"Unique values in output: {np.unique(output)}")
#     print(f"Unique values in truth: {np.unique(truth)}")

#     output = output.flatten()
#     truth = np.array(truth).flatten()
#     # Define the positive class you are interested in, e.g., 2
#     positive_class = 2
#     negative_class = 1

#     ad_results_dict["TP"] = np.sum(
#         (output == positive_class) & (truth == positive_class)
#     )
#     ad_results_dict["TN"] = np.sum(
#         (output == negative_class) & (truth == negative_class)
#     )
#     ad_results_dict["FP"] = np.sum(
#         (output == positive_class) & (truth == negative_class)
#     )
#     ad_results_dict["FN"] = np.sum(
#         (output == negative_class) & (truth == positive_class)
#     )

#     print(
#         f"TP: {ad_results_dict['TP']}, TN: {ad_results_dict['TN']}, FP: {ad_results_dict['FP']}, FN: {ad_results_dict['FN']}"
#     )

#     # AUC and % of detected anomalies
#     detected_anomalies = np.sum((output == 2) & (truth == 2)) / np.sum(truth == 2)
#     print(f"Detected anomalies: {detected_anomalies}")
#     ad_results_dict["detected_anomalies"] = detected_anomalies
#     # AUC
#     # Compute the ROC curve
#     from sklearn.metrics import roc_curve, auc

#     fpr, tpr, _ = roc_curve(truth, output, pos_label=positive_class)
#     roc_auc = auc(fpr, tpr)
#     print(f"AUC: {roc_auc}")
#     ad_results_dict["AUC"] = roc_auc

#     return ad_results_dict
