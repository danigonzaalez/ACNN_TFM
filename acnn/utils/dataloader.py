import numpy as np
import torch
from PIL import Image
import os

def read_rawPV(filename):
    """
    Reads the header of a hyperspectral image in RAW Pixel-Vector format (FiveBillionPixels)
    and returns a memory-mapped array (np.memmap) to avoid loading the entire file into RAM.
    """
    print(f"* Memory-mapping RAW PV image: {filename}")
    
    # Read only the first 3 integers (12 bytes) to get the dimensions
    header = np.fromfile(filename, dtype='int32', count=3)
    B = header[0]
    H = header[1]
    V = header[2]
    
    # Create a memory map for the data, skipping the 12-byte header.
    # This acts as a normal NumPy array, but loads data from disk on the fly.
    data_memmap = np.memmap(filename, dtype='int32', mode='r', offset=12, shape=(V, H, B))
    
    return (data_memmap, H, V, B)

def read_grayscale_mask(filename):
    """
    Reads the ground truth mask in grayscale.
    """
    print(f"* Reading GT mask: {filename}")
    mask = Image.open(filename).convert('L')
    data = np.array(mask, dtype=np.uint8)
    H, V = mask.size
    return (data, H, V)

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