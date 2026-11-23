import glob
import os
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

class PreextractedPatchDataset(Dataset):
    """
    Dataset that loads pre-extracted image patches (*_img.npy) and mask patches
    (*_mask.npy) from disk for fast semantic segmentation training and evaluation.
    """
    def __init__(self, patches_dir, transform_flag=False):
        self.patches_dir = patches_dir
        self.transform_flag = transform_flag

        self.img_paths = sorted(glob.glob(os.path.join(patches_dir, "*_img.npy")))
        self.mask_paths = sorted(glob.glob(os.path.join(patches_dir, "*_mask.npy")))

        if len(self.img_paths) == 0:
            raise ValueError(f"No '*_img.npy' patch files found in directory: {patches_dir}")
        if len(self.img_paths) != len(self.mask_paths):
            raise ValueError(
                f"Mismatch between number of image patches ({len(self.img_paths)}) "
                f"and mask patches ({len(self.mask_paths)}) in {patches_dir}"
            )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load pre-normalized patch (C, P, P) and remapped mask (P, P)
        img_np = np.load(self.img_paths[idx])
        mask_np = np.load(self.mask_paths[idx])

        img_tensor = torch.from_numpy(img_np).float()
        mask_tensor = torch.from_numpy(mask_np).long()

        # Joint transformations (Image + Mask)
        if self.transform_flag:
            if random.random() > 0.5:
                img_tensor = torch.flip(img_tensor, dims=[2])  # Horizontal Flip (W)
                mask_tensor = torch.flip(mask_tensor, dims=[1])
            if random.random() > 0.5:
                img_tensor = torch.flip(img_tensor, dims=[1])  # Vertical Flip (H)
                mask_tensor = torch.flip(mask_tensor, dims=[0])

        return img_tensor, mask_tensor


# -----------------------------------------------------------------
# PYTORCH - UTILS
# -----------------------------------------------------------------

def compute_band_percentiles(image_paths, H=6908, V=7300, B=4, q_low=1.0, q_high=99.0, max_pixels_per_image=800_000):
    """
    Computes per-band (q_low, q_high) percentiles over training images to avoid data leakage.
    Samples pixels per full image to keep memory usage and runtime reasonable.
    """
    rng = np.random.default_rng(42)
    samples_per_band = [[] for _ in range(B)]

    for img_path in image_paths:
        raw_data = np.fromfile(img_path, dtype="int32", offset=12)
        full_image = raw_data.reshape((V, H, B))

        flat_img = full_image.reshape(-1, B)
        n_px = flat_img.shape[0]
        if n_px > max_pixels_per_image:
            idx = rng.choice(n_px, size=max_pixels_per_image, replace=False)
            sampled = flat_img[idx]
        else:
            sampled = flat_img

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