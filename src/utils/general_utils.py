import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


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


def generate_rgb_map(output, truth, H, V):
    """
    Generates the RGB classification map based on FiveBillionPixels colors.
    """
    # Label to color mapping (based on the dataset README)
    color_map = {
        0: [0, 0, 0],       # unlabeled
        1: [200, 0, 0],     # industrial area
        2: [0, 200, 0],     # paddy field
        3: [150, 250, 0],   # irrigated field
        4: [150, 200, 150], # dry cropland
        5: [200, 0, 200],   # garden land
        6: [150, 0, 250],   # arbor forest
        7: [150, 150, 250], # shrub forest
        8: [200, 150, 200], # park
        9: [250, 200, 0],   # natural meadow
        10: [200, 200, 0],  # artificial meadow
        11: [0, 0, 200],    # river
        12: [250, 0, 150],  # urban residential
        13: [0, 150, 200],  # lake
        14: [0, 200, 250],  # pond
        15: [150, 200, 250],# fish pond
        16: [250, 250, 250],# snow
        17: [200, 200, 200],# bareland
        18: [200, 150, 150],# rural residential
        19: [250, 200, 150],# stadium
        20: [150, 150, 0],  # square
        21: [250, 150, 150],# road
        22: [250, 150, 0],  # overpass
        23: [250, 200, 250],# railway station
        24: [200, 150, 0]   # airport
    }
    
    output_rgb = np.zeros((V, H, 3), dtype=np.uint8)
    for idx in range(H * V):
        i, j = divmod(idx, H)
        pred_class = output[idx]
        output_rgb[i, j] = color_map.get(pred_class, [0, 0, 0])
        
    return Image.fromarray(output_rgb, 'RGB')