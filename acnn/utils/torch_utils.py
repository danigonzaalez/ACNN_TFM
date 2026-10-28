import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
import math

# ----------------
# GLOBALS
# ----------------
AUM = 1 # Data augmentation enabled by default (1 = ON, 0 = OFF)

def select_patch(data, H, V, sizex, sizey, x, y):
    """
    Extracts a patch of size (sizex, sizey) around the center pixel (x, y).
    Reads directly from the memory-mapped array, normalizes (16-bit), and reorders bands.
    """
    x1 = int(x) - int(sizex / 2)
    x2 = int(x) + int(math.ceil(sizex / 2))
    y1 = int(y) - int(sizey / 2)
    y2 = int(y) + int(math.ceil(sizey / 2))
    
    # Boundary padding/shifting
    if x1 < 0:
        x1 = 0
        x2 = sizex
    if x2 > H:
        x2 = H
        x1 = H - sizex
    if y1 < 0:
        y1 = 0
        y2 = sizey
    if y2 > V:
        y2 = V
        y1 = V - sizey

    # Extract patch from memory-mapped array (V, H, B)
    patch = data[y1:y2, x1:x2, :]
    
    # Hardcoded 16-bit logic: Band reordering and Normalization
    # Original 16 Bits: [B0:Blue, B1:Green, B2:Red, B3:NIR] -> [Red, Green, Blue, NIR]
    patch = patch[:, :, [2, 1, 0, 3]]
    patch = patch.astype(np.float32) / 65535.0
        
    # Convert to PyTorch Tensor and permute to (C, H, W)
    patch = torch.FloatTensor(patch).permute(2, 0, 1)
    
    return patch


def select_training_samples(truth, H, V, sizex, sizey, percentages):
    """
    Generates index lists for training, validation, and testing samples
    based on the ground truth mask purely at the pixel level.
    """
    print("* Selecting training samples (Pixel-level)")
    num_classes = np.max(truth)
    print("  Num classes:", num_classes)
    
    # Initialize lists for each class
    class_lists = [[] for _ in range(num_classes)]
    
    # Avoid edges where patches cannot be fully extracted
    xmin = int(sizex / 2)
    xmax = H - int(math.ceil(sizex / 2))
    ymin = int(sizey / 2)
    ymax = V - int(math.ceil(sizey / 2))
    
    # Identify valid pixels for each class across the whole image
    for ind in range(H * V):
        i = ind // H # Row
        j = ind % H  # Col
        label = truth[i, j]   # truth is a 2D matrix of VxH
        
        # Only use pixels where a full patch can fit and label > 0
        if ymin <= i <= ymax and xmin <= j <= xmax and label > 0:
            class_lists[label - 1].append(ind)
            
    # Shuffle pixels within each class
    for i in range(num_classes):
        random.shuffle(class_lists[i])
        
    train_idx = []
    val_idx = []
    test_idx = []
    non_empty_classes = 0
    
    print("  Class  # :   total | train |   val |    test")
    for i in range(num_classes):
        total_samples = len(class_lists[i])
        
        # Calculate samples for train (tot0), val (tot1), and test (tot2)
        tot0 = percentages[0] if percentages[0] >= 1 else int(percentages[0] * total_samples)
        tot1 = percentages[1] if percentages[1] >= 1 else int(percentages[1] * total_samples)
        
        if len(percentages) > 2:   # if samples for testing were provided
            tot2 = percentages[2] if percentages[2] >= 1 else int(percentages[2] * total_samples)
        else:   # if not provided samples for testing, take the rest
            tot2 = total_samples - tot0 - tot1
            
        if tot0 >= total_samples: tot0 = total_samples // 3
        if tot0 > 0: non_empty_classes += 1
        if tot1 >= total_samples: tot1 = total_samples // 3
        if tot2 >= total_samples: tot2 = total_samples // 3
        
        for j in range(total_samples):
            if j < tot0:
                train_idx.append(class_lists[i][j])
            elif j < (tot0 + tot1):
                val_idx.append(class_lists[i][j])
            elif j < (tot0 + tot1 + tot2):
                test_idx.append(class_lists[i][j])
                
        print(f"  Class {i+1:2d} : {total_samples:7d} | {tot0:5d} | {tot1:5d} | {tot2:7d}")
        
    return train_idx, val_idx, test_idx, num_classes, non_empty_classes


# -----------------------------------------------------------------
# PYTORCH - DATASETS
# -----------------------------------------------------------------

class HyperDataset(Dataset):
    """
    Dataset class that retrieves image patches dynamically from a memory-mapped array.
    """
    def __init__(self, data, truth, samples, H, V, sizex, sizey):
        self.data = data
        self.truth = truth
        self.samples = samples
        self.H = H
        self.V = V
        self.sizex = sizex
        self.sizey = sizey
        
        # Data Augmentation specific for aerial/satellite imagery
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx] % self.H   # column (x)
        y = self.samples[idx] // self.H   # row (y)
        
        patch = select_patch(self.data, self.H, self.V, self.sizex, self.sizey, x, y)
        
        if AUM == 1:
            patch = self.transform(patch)
            
        # Neural network outputs 0 to N-1, so we subtract 1 from the truth label
        target_label = self.truth[y, x] - 1
        
        return patch, target_label


# -----------------------------------------------------------------
# PYTORCH - UTILS
# -----------------------------------------------------------------

def update_lr(optimizer, lr):
    """Updates the learning rate for the optimizer."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def accuracy_mean_deviation(OA, AA, aa):
    """Calculates the mean and standard deviation for the experiments."""
    n = len(OA)
    nclases = len(aa[0])
    print(f"* Means and deviations ({n} exp):")
    if n == 0:
        return (0, 0, 0, 0, [0] * nclases, [0] * nclases)
    elif n == 1:
        return (OA[0], AA[0], 0, 0, aa[0], [0] * nclases)
    
    # Means
    OAm = sum(OA) / n
    AAm = sum(AA) / n
    aam = [sum(aa[i][j] for i in range(n)) / n for j in range(nclases)]
    
    # Deviations (n-1)
    OAd = math.sqrt(sum((OA[i] - OAm)**2 for i in range(n)) / (n - 1)) if n > 1 else 0
    AAd = math.sqrt(sum((AA[i] - AAm)**2 for i in range(n)) / (n - 1)) if n > 1 else 0
    aad = [math.sqrt(sum((aa[i][j] - aam[j])**2 for i in range(n)) / (n - 1)) if n > 1 else 0 for j in range(nclases)]
    
    for j in range(1, nclases):
        print(f"  Class {j:02d}: {aam[j]:02.02f}±{aad[j]:02.02f}")
    print(f"  OA={OAm:02.02f}±{OAd:02.02f}, AA={AAm:02.02f}±{AAd:02.02f}")
    
    return (OAm, AAm, OAd, AAd, aam, aad)