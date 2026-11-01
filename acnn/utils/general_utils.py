import os
import json
import numpy as np
import matplotlib.pyplot as plt


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


def plot_learning_curves(history, output_dir, exp_number):
    """
    Plots and saves the training and validation learning curves.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # 1. Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    if len(history['val_loss']) > 0:
        plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    plt.title('Loss Evolution')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 2. Plot Validation Accuracy
    if len(history['val_acc']) > 0:
        plt.subplot(1, 2, 2)
        plt.plot(epochs, history['val_acc'], 'g-', label='Val Accuracy')
        plt.title('Accuracy Evolution')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'learning_curves_exp{exp_number}.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()

def print_global_sampling_summary(train_stats, test_stats, train_idx, val_idx, test_idx, nclases):
    """
    Prints a formatted summary table showing the number of patches/idx extracted per class.
    """
    print("* Global sampling summary:")
    print(f"  Num classes: {nclases}")
    print("  Class  # :      total | train |   val |    test")
    
    for c in range(1, 25):
        tot = train_stats['total'][c] + test_stats['total'][c]
        tr = train_stats['s1'][c]
        vl = train_stats['s2'][c]
        ts = test_stats['s1'][c]
        
        # Uncomment the 'if' condition below if you want to hide classes with zero samples
        # if tot > 0:
        print(f"  Class {c:2d} : {tot:10d} | {tr:5d} | {vl:5d} | {ts:7d}")
            
    print(f"\n  -> Final Train patches: {len(train_idx)}")
    print(f"  -> Final Val patches:   {len(val_idx)}")
    print(f"  -> Final Test patches:  {len(test_idx)}\n")