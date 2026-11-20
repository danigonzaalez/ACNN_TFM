import matplotlib.pyplot as plt
import numpy as np
import json
from datetime import datetime
import argparse

def plot_results(filename, config=None):
    ''' Function to plot the results of the models
        args:   filename: name of the file with the results
                config: configuration dictionary
        return: None
    '''
    # Read the results from the file
    with open(filename, 'r') as f:
        results = json.load(f)

    # Get the timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # Generate dynamic colors based on the number of models
    models = list(results.keys())
    colors = plt.cm.tab10(range(len(models)))  # Generate colors dynamically

    # Create a grid of plots
    fig, ax = plt.subplots(2, 2, figsize=(14, 12))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)  # Adjust spacing

    # Plot Overall Accuracy (OA)
    for i, (model, result) in enumerate(results.items()):
        bar = ax[0, 0].bar(model, result['OA'], yerr=result.get('OA_dev', 0), color=colors[i], capsize=5)
        ax[0, 0].bar_label(bar, labels=[f"{result['OA']:.2f}"], padding=3)
    ax[0, 0].set_ylabel('Overall Accuracy', fontsize=12)
    ax[0, 0].set_title('Overall Accuracy of the Models', fontsize=14)

    # Plot Average Accuracy (AA)
    for i, (model, result) in enumerate(results.items()):
        bar = ax[0, 1].bar(model, result['AA'], yerr=result.get('AA_dev', 0), color=colors[i], capsize=5)
        ax[0, 1].bar_label(bar, labels=[f"{result['AA']:.2f}"], padding=3)
    ax[0, 1].set_ylabel('Average Accuracy', fontsize=12)
    ax[0, 1].set_title('Average Accuracy of the Models', fontsize=14)

    # Filter out the "Unknown" class for per class results
    filtered_classes = ["Non anomaly", "Anomaly"]
    filtered_class_indices = [1, 2]  # Indices for Non anomaly and Anomaly

    # Plot per class results as a line plot without "Unknown"
    for i, (model, result) in enumerate(results.items()):
        filtered_accuracies = [result['class_aa'][j] for j in filtered_class_indices]
        filtered_std_devs = [result['class_aa_dev'][j] for j in filtered_class_indices] if 'class_aa_dev' in result else [0, 0]
        ax[1, 0].errorbar(filtered_classes, filtered_accuracies, yerr=filtered_std_devs, label=model, color=colors[i], marker='o', linestyle='--', capsize=5)
        for j, acc in enumerate(filtered_accuracies):
            ax[1, 0].text(j, acc + 1, f"{acc:.2f}", ha='center', va='bottom', fontsize=10)
    ax[1, 0].set_ylabel('Per Class Accuracy', fontsize=12)
    ax[1, 0].set_title('Per Class Accuracy of the Models', fontsize=14)
    ax[1, 0].set_xticks(range(len(filtered_classes)))
    ax[1, 0].set_xticklabels(filtered_classes, rotation=0, fontsize=10)
    ax[1, 0].legend()

    # Create grouped bar chart for Per Class Accuracy without "Unknown"
    bar_width = 0.5 / len(results)
    index = np.arange(len(filtered_classes))
    
    for i, (model, result) in enumerate(results.items()):
        filtered_values = [result['class_aa'][j] for j in filtered_class_indices]
        filtered_std_devs = [result['class_aa_dev'][j] for j in filtered_class_indices] if 'class_aa_dev' in result else [0, 0]
        ax[1, 1].bar(index + i * bar_width, filtered_values, bar_width, yerr=filtered_std_devs, label=model, color=colors[i], capsize=5)

    ax[1, 1].set_title("Per Class Accuracy (Grouped Bar Chart)", fontsize=14)
    ax[1, 1].set_xticks(index + bar_width * (len(results) - 1) / 2)
    ax[1, 1].set_xticklabels(filtered_classes, fontsize=10)
    ax[1, 1].set_ylabel("Accuracy", fontsize=12)
    ax[1, 1].legend()

    # Add configuration dictionary to the plot
    if config is not None:
        config_text = json.dumps(config)
        fig.text(0.5, 0.005, config_text, ha='center', va='bottom', fontsize=6, wrap=True, bbox=dict(facecolor='white', alpha=0.5))
    # Save the plot
    #if results folder exist, save the plot there
    try:
        plt.savefig(f"results/results_{timestamp}.png")
    except:
        plt.savefig(f"results_{timestamp}.png")
    #plt.show()

    # Generate a summary of the results in latex tabular format
    print("\nSummary of the results:")
    print("Model\t\t\tOA\t\tAA")
    for model, result in results.items():
        oa_dev = result.get('OA_dev', 0)
        aa_dev = result.get('AA_dev', 0)
        print(f"{model:20}\t{result['OA']:<5.2f}±{oa_dev:<0.2f}\t{result['AA']:<5.2f}±{aa_dev:<0.2f}")
    
    print("\nPer Class Accuracies:")
    print("Model\t\t\tNon anomaly\tAnomaly")
    for model, result in results.items():
        non_anomaly_acc = result['class_aa'][1]
        anomaly_acc = result['class_aa'][2]
        non_anomaly_dev = result['class_aa_dev'][1] if 'class_aa_dev' in result else 0
        anomaly_dev = result['class_aa_dev'][2] if 'class_aa_dev' in result else 0
        print(f"{model:20}\t{non_anomaly_acc:<5.2f}±{non_anomaly_dev:<0.2f}\t{anomaly_acc:<5.2f}±{anomaly_dev:<0.2f}")



    print("\nAnomaly Detection Results:")
    # "AD_results": {
    #     "TP": {
    #         "mean": 110846.0,
    #         "std_dev": 16913.0
    #     },
    #     "TN": {
    #         "mean": 1699717.0,
    #         "std_dev": 54655.0
    #     },
    #     "FP": {
    #         "mean": 119653.0,
    #         "std_dev": 54655.0
    #     },
    #     "FN": {
    #         "mean": 57333.0,
    #         "std_dev": 16913.0
    #     },
    #     "detected_anomalies": {
    #         "mean": 0.6577968203856128,
    #         "std_dev": 0.10036733507011408
    #     },
    #     "AUC": {
    #         "mean": 0.7955017314223505,
    #         "std_dev": 0.03512275866530162
    #     }
    print("Model\t\t\tTP\t\t\tTN\t\t\tFP\t\t\tFN\t\t\tDetected Anomalies\tAUC")
    for model, result in results.items():
        if "AD_results" in result.keys():
            ad_results = result["AD_results"]
            tp_mean = ad_results["TP"]["mean"]
            tp_std_dev = ad_results["TP"]["std_dev"]
            tn_mean = ad_results["TN"]["mean"]
            tn_std_dev = ad_results["TN"]["std_dev"]
            fp_mean = ad_results["FP"]["mean"]
            fp_std_dev = ad_results["FP"]["std_dev"]
            fn_mean = ad_results["FN"]["mean"]
            fn_std_dev = ad_results["FN"]["std_dev"]
            detected_anomalies_mean = ad_results["detected_anomalies"]["mean"]
            detected_anomalies_std_dev = ad_results["detected_anomalies"]["std_dev"]
            auc_mean = ad_results["AUC"]["mean"]
            auc_std_dev = ad_results["AUC"]["std_dev"]
            print(f"{model:20}\t{tp_mean:<10.2f}±{tp_std_dev:<5.2f}\t{tn_mean:<10.2f}±{tn_std_dev:<5.2f}\t{fp_mean:<10.2f}±{fp_std_dev:<5.2f}\t{fn_mean:<10.2f}±{fn_std_dev:<5.2f}\t{detected_anomalies_mean:<0.4f}±{detected_anomalies_std_dev:<0.2f}\t\t{auc_mean:<0.4f}±{auc_std_dev:<0.2f}")

    print("\n\nPlotting the comparison of Anomaly Detection results...")

    # Filter models that have AD_results
    ad_models = [model for model in models if "AD_results" in results[model]]

    # Create subplots for each metric
    fig, axs = plt.subplots(3, 2, figsize=(14, 18))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)  # Adjust spacing

    # Plot True Positives (TP) comparison
    for i, model in enumerate(ad_models):
        tp_mean = results[model]['AD_results']['TP']['mean']
        tp_std = results[model]['AD_results']['TP']['std_dev']
        bar = axs[0, 0].bar(model, tp_mean, yerr=tp_std, capsize=5, color=colors[i])
        axs[0, 0].bar_label(bar, labels=[f"{tp_mean:.2f}"], padding=3)
    axs[0, 0].set_title('True Positives')
    axs[0, 0].set_ylabel('Count')

    # Plot True Negatives (TN) comparison
    for i, model in enumerate(ad_models):
        tn_mean = results[model]['AD_results']['TN']['mean']
        tn_std = results[model]['AD_results']['TN']['std_dev']
        bar = axs[0, 1].bar(model, tn_mean, yerr=tn_std, capsize=5, color=colors[i])
        axs[0, 1].bar_label(bar, labels=[f"{tn_mean:.2f}"], padding=3)
    axs[0, 1].set_title('True Negatives')
    axs[0, 1].set_ylabel('Count')

    # Plot False Positives (FP) comparison
    for i, model in enumerate(ad_models):
        fp_mean = results[model]['AD_results']['FP']['mean']
        fp_std = results[model]['AD_results']['FP']['std_dev']
        bar = axs[1, 0].bar(model, fp_mean, yerr=fp_std, capsize=5, color=colors[i])
        axs[1, 0].bar_label(bar, labels=[f"{fp_mean:.2f}"], padding=3)
    axs[1, 0].set_title('False Positives')
    axs[1, 0].set_ylabel('Count')

    # Plot False Negatives (FN) comparison
    for i, model in enumerate(ad_models):
        fn_mean = results[model]['AD_results']['FN']['mean']
        fn_std = results[model]['AD_results']['FN']['std_dev']
        bar = axs[1, 1].bar(model, fn_mean, yerr=fn_std, capsize=5, color=colors[i])
        axs[1, 1].bar_label(bar, labels=[f"{fn_mean:.2f}"], padding=3)
    axs[1, 1].set_title('False Negatives')
    axs[1, 1].set_ylabel('Count')

    # Plot Detected Anomalies comparison
    for i, model in enumerate(ad_models):
        da_mean = results[model]['AD_results']['detected_anomalies']['mean']
        da_std = results[model]['AD_results']['detected_anomalies']['std_dev']
        bar = axs[2, 0].bar(model, da_mean, yerr=da_std, capsize=5, color=colors[i])
        axs[2, 0].bar_label(bar, labels=[f"{da_mean:.4f}"], padding=3)
    axs[2, 0].set_title('Detected Anomalies')
    axs[2, 0].set_ylabel('Proportion')

    # Plot AUC comparison
    for i, model in enumerate(ad_models):
        auc_mean = results[model]['AD_results']['AUC']['mean']
        auc_std = results[model]['AD_results']['AUC']['std_dev']
        bar = axs[2, 1].bar(model, auc_mean, yerr=auc_std, capsize=5, color=colors[i])
        axs[2, 1].bar_label(bar, labels=[f"{auc_mean:.4f}"], padding=3)
    axs[2, 1].set_title('AUC')
    axs[2, 1].set_ylabel('Score')

    for ax in axs.flat:
        ax.set_xticklabels(ad_models, rotation=0, ha='right')

    plt.tight_layout()
    plt.savefig(f"results/ad_comparison_{timestamp}.png")
    #plt.show()



if __name__ == '__main__':
    
    # Argument parser to get the filename
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="Name of the file to plot")
    parser.add_argument("--config", help="Name of the configuration file", default=None)
    args = parser.parse_args()
    filename = args.filename
    config = args.config
    if config is not None:
        config = json.load(open(config))

    plot_results(filename, config)
