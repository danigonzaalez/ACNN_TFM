import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms
import math
import random
from PIL import Image

# -----------------------------------------------------------------
# PYTORCH - DATASETS
# -----------------------------------------------------------------

class DenseQuarterDataset(Dataset):
    """
    Dataset that loads hyperspectral images into RAM and divides them into 4 quarters.
    Returns dense images (C, H, W) and dense masks (H, W) for semantic segmentation.
    """
    def __init__(self, image_paths, mask_paths, H=6908, V=7300, B=4, transform_flag=False, p_low=None, p_high=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.H = H
        self.V = V
        self.B = B
        self.transform_flag = transform_flag
        self.p_low = p_low
        self.p_high = p_high
        
        self.num_images = len(image_paths)
        self.num_quarters = 4
        
        # Optional: Spatial data augmentation (flip) adapted for dense prediction
        # In segmentation, if you flip the image, you MUST flip the mask exactly the same way.
        # We will handle this manually in __getitem__ to ensure consistency.

    def __len__(self):
        # 4 samples per image
        return self.num_images * self.num_quarters

    def __getitem__(self, idx):
        # Determine which image and which quarter the index corresponds to
        img_idx = idx // self.num_quarters
        quarter_idx = idx % self.num_quarters
        
        # Read entire RAW image into RAM (skipping the 12-byte header)
        raw_data = np.fromfile(self.image_paths[img_idx], dtype='int32', offset=12)
        full_image = raw_data.reshape((self.V, self.H, self.B))
        
        # Read entire mask
        full_mask = np.array(Image.open(self.mask_paths[img_idx]), dtype=np.int64)

        # Correct mask orientation if dimensions are swapped
        if full_mask.shape == (self.H, self.V):
            full_mask = full_mask.T

        # Check spatial dimensions
        if full_image.shape[:2] != full_mask.shape:
            raise ValueError(
                f"Dimension mismatch: RAW={full_image.shape[:2]}, "
                f"MASK={full_mask.shape}"
            )
        
        # Calculate quarter coordinates (quarter_idx: 0=Top-Left, 1=Top-Right, 2=Bot-Left, 3=Bot-Right)
        mid_H, mid_V = self.H // 2, self.V // 2
        
        if quarter_idx == 0:
            img_q = full_image[:mid_V, :mid_H, :]
            mask_q = full_mask[:mid_V, :mid_H]
        elif quarter_idx == 1:
            img_q = full_image[:mid_V, mid_H:, :]
            mask_q = full_mask[:mid_V, mid_H:]
        elif quarter_idx == 2:
            img_q = full_image[mid_V:, :mid_H, :]
            mask_q = full_mask[mid_V:, :mid_H]
        else:
            img_q = full_image[mid_V:, mid_H:, :]
            mask_q = full_mask[mid_V:, mid_H:]
            
        # Percentile scaling (p1-p99 from train) clipped to [0.0, 1.0] + band reorder
        img_q_float = img_q.astype(np.float32)
        if self.p_low is not None and self.p_high is not None:
            scale = self.p_high - self.p_low
            img_q_float = (img_q_float - self.p_low) / scale
            img_q_float = np.clip(img_q_float, 0.0, 1.0)
        else:
            img_q_float = img_q_float / 65535.0

        img_q_norm = img_q_float[:, :, [2, 1, 0, 3]]
        
        # Convert to tensors
        img_tensor = torch.FloatTensor(img_q_norm).permute(2, 0, 1)  # (C, H, W)
        # (H, W). Classes 1-24 become 0-23, and class 0 becomes 255 (ignore_index)
        mask_tensor = torch.LongTensor(mask_q)
        mask_tensor = torch.where(mask_tensor == 0, torch.tensor(255, dtype=torch.long), mask_tensor - 1)

        # Pad the image and mask to ensure spatial dimensions (V, H) are multiples of 32
        _, v_curr, h_curr = img_tensor.shape
        
        pad_v = (32 - (v_curr % 32)) % 32
        pad_h = (32 - (h_curr % 32)) % 32
        
        if pad_v > 0 or pad_h > 0:
            # F.pad expects (pad_left, pad_right, pad_top, pad_bottom)
            # We apply pad_h to the right and pad_v to the bottom
            img_tensor = F.pad(img_tensor, (0, pad_h, 0, pad_v), mode='constant', value=0.0)
            mask_tensor = F.pad(mask_tensor, (0, pad_h, 0, pad_v), mode='constant', value=255)
        
        # Joint transformations (Image + Mask)
        if self.transform_flag:
            if random.random() > 0.5:
                img_tensor = torch.flip(img_tensor, dims=[2]) # Horizontal Flip (W)
                mask_tensor = torch.flip(mask_tensor, dims=[1])
            if random.random() > 0.5:
                img_tensor = torch.flip(img_tensor, dims=[1]) # Vertical Flip (H)
                mask_tensor = torch.flip(mask_tensor, dims=[0])
                
        return img_tensor, mask_tensor


# -----------------------------------------------------------------
# PYTORCH - UTILS
# -----------------------------------------------------------------

def compute_band_percentiles(image_paths, H=6908, V=7300, B=4, q_low=1.0, q_high=99.0, max_pixels_per_quarter=200_000):
    """
    Computes per-band (q_low, q_high) percentiles over training images to avoid data leakage.
    Samples pixels per quarter to keep memory usage and runtime reasonable.
    """
    rng = np.random.default_rng(42)
    samples_per_band = [[] for _ in range(B)]
    mid_H, mid_V = H // 2, V // 2

    for img_path in image_paths:
        raw_data = np.fromfile(img_path, dtype="int32", offset=12)
        full_image = raw_data.reshape((V, H, B))

        quarters = [
            full_image[:mid_V, :mid_H, :],
            full_image[:mid_V, mid_H:, :],
            full_image[mid_V:, :mid_H, :],
            full_image[mid_V:, mid_H:, :],
        ]

        for q in quarters:
            flat_q = q.reshape(-1, B)
            n_px = flat_q.shape[0]
            if n_px > max_pixels_per_quarter:
                idx = rng.choice(n_px, size=max_pixels_per_quarter, replace=False)
                sampled = flat_q[idx]
            else:
                sampled = flat_q

            for b in range(B):
                samples_per_band[b].append(sampled[:, b])

    p_low = np.zeros(B, dtype=np.float32)
    p_high = np.zeros(B, dtype=np.float32)

    for b in range(B):
        band_vals = np.concatenate(samples_per_band[b], axis=0)
        p_low[b] = np.percentile(band_vals, q_low)
        p_high[b] = np.percentile(band_vals, q_high)
        # Avoid division by zero if p_high == p_low
        if p_high[b] <= p_low[b]:
            p_high[b] = p_low[b] + 1.0

    return p_low, p_high

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