import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
import math
from PIL import Image

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

def build_global_samples(mask_paths, H, V, sizex, sizey, samples_train_per_img, samples_val_per_img):
    """
    Iterates over all masks efficiently and extracts a fixed number of samples per class per image.
    Returns global lists of tuples: (img_idx, x, y, label)
    """
    print(f"* Building global dataset indexing from {len(mask_paths)} images...")
    train_samples = []
    val_samples = []

    xmin, xmax = int(sizex / 2), H - int(math.ceil(sizex / 2))
    ymin, ymax = int(sizey / 2), V - int(math.ceil(sizey / 2))

    for img_idx, mask_path in enumerate(mask_paths):
        # Read mask using PIL and convert to numpy
        mask = np.array(Image.open(mask_path))
        
        # Avoid borders and empty pixels using fast numpy vectorization
        valid_mask = (mask > 0)
        valid_mask[:ymin, :] = False
        valid_mask[ymax:, :] = False
        valid_mask[:, :xmin] = False
        valid_mask[:, xmax:] = False

        # Get all valid coordinates
        y_coords, x_coords = np.where(valid_mask)
        labels = mask[y_coords, x_coords]

        # Group by class (assuming max 24 classes as per dataset spec)
        class_lists = [[] for _ in range(24)]
        for y, x, label in zip(y_coords, x_coords, labels):
            class_lists[label - 1].append((img_idx, x, y, label))

        # Sample limits per image
        for c in range(24):
            if len(class_lists[c]) > 0:
                random.shuffle(class_lists[c])
                t_samples = class_lists[c][:samples_train_per_img]
                v_samples = class_lists[c][samples_train_per_img : samples_train_per_img + samples_val_per_img]
                
                train_samples.extend(t_samples)
                val_samples.extend(v_samples)

    # Global shuffle to mix patches from all images
    random.shuffle(train_samples)
    random.shuffle(val_samples)
    
    print(f"  -> Total Global Train patches: {len(train_samples)}")
    print(f"  -> Total Global Val patches: {len(val_samples)}")
    return train_samples, val_samples


# -----------------------------------------------------------------
# PYTORCH - DATASETS
# -----------------------------------------------------------------

class MultiImageDataset(Dataset):
    """
    Dataset that retrieves patches dynamically from multiple memory-mapped arrays.
    """
    def __init__(self, image_paths, samples, H, V, sizex, sizey, transform_flag=False):
        self.image_paths = image_paths
        self.samples = samples
        self.H = H
        self.V = V
        self.sizex = sizex
        self.sizey = sizey
        self.transform_flag = transform_flag
        
        # Initialize memmaps for all images once (acts as virtual pointers, takes ~0 RAM)
        self.memmaps = []
        for path in image_paths:
            # We assume 4 bands, offset 12 for the header
            mm = np.memmap(path, dtype='int32', mode='r', offset=12, shape=(V, H, 4))
            self.memmaps.append(mm)
        
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Extract tuple info
        img_idx, x, y, label = self.samples[idx]
        
        # Select the correct memory map for this patch
        data = self.memmaps[img_idx]
        
        # Call your existing select_patch function
        patch = select_patch(data, self.H, self.V, self.sizex, self.sizey, x, y)
        
        if self.transform_flag:
            patch = self.transform(patch)
            
        # Target label (0-indexed for PyTorch)
        target_label = label - 1
        
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

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=7, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.mode = mode

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif (self.mode == 'min' and val_loss > self.best_loss - self.min_delta) or \
             (self.mode == 'max' and val_loss < self.best_loss + self.min_delta):
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0