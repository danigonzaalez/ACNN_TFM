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

class DensePatchDataset(Dataset):
    """
    Dataset that loads hyperspectral images into RAM, applies bottom/right padding
    so spatial dimensions are divisible by patch_size, and extracts regular
    (patch_size x patch_size) patches and masks for semantic segmentation.
    """
    def __init__(
        self,
        image_paths,
        mask_paths,
        H=6908,
        V=7300,
        B=4,
        patch_size=512,
        transform_flag=False,
        p_low=None,
        p_high=None,
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.H = int(H)
        self.V = int(V)
        self.B = int(B)
        self.patch_size = int(patch_size)
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be > 0, got {self.patch_size}")

        self.transform_flag = transform_flag
        self.p_low = p_low
        self.p_high = p_high

        self.num_images = len(image_paths)

        # Compute bottom/right padding so V and H are exact multiples of patch_size
        self.pad_v = (self.patch_size - (self.V % self.patch_size)) % self.patch_size
        self.pad_h = (self.patch_size - (self.H % self.patch_size)) % self.patch_size
        self.v_pad_total = self.V + self.pad_v
        self.h_pad_total = self.H + self.pad_h

        self.n_rows = self.v_pad_total // self.patch_size
        self.n_cols = self.h_pad_total // self.patch_size
        self.patches_per_img = self.n_rows * self.n_cols

    def __len__(self):
        return self.num_images * self.patches_per_img

    def __getitem__(self, idx):
        # Determine image index and patch grid coordinates
        img_idx = idx // self.patches_per_img
        patch_idx = idx % self.patches_per_img
        row = patch_idx // self.n_cols
        col = patch_idx % self.n_cols

        y0 = row * self.patch_size
        x0 = col * self.patch_size
        y1 = y0 + self.patch_size
        x1 = x0 + self.patch_size

        # Read entire RAW image into RAM (skipping the 12-byte header)
        raw_data = np.fromfile(self.image_paths[img_idx], dtype="int32", offset=12)
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

        # Percentile scaling (p1-p99 from train) clipped to [0.0, 1.0] + band reorder (B, G, R, NIR -> R, G, B, NIR)
        img_float = full_image.astype(np.float32)
        if self.p_low is not None and self.p_high is not None:
            scale = self.p_high - self.p_low
            img_float = (img_float - self.p_low) / scale
            img_float = np.clip(img_float, 0.0, 1.0)
        else:
            img_float = img_float / 65535.0

        img_norm = img_float[:, :, [2, 1, 0, 3]]

        # Apply bottom/right padding if needed so edge patches have shape (patch_size, patch_size)
        if self.pad_v > 0 or self.pad_h > 0:
            img_norm = np.pad(
                img_norm,
                ((0, self.pad_v), (0, self.pad_h), (0, 0)),
                mode="constant",
                constant_values=0.0,
            )
            full_mask = np.pad(
                full_mask,
                ((0, self.pad_v), (0, self.pad_h)),
                mode="constant",
                constant_values=0,
            )

        # Extract patch (P, P, C) and mask (P, P)
        img_p = img_norm[y0:y1, x0:x1, :]
        mask_p = full_mask[y0:y1, x0:x1]

        # Convert to tensors: (C, P, P) and (P, P)
        img_tensor = torch.from_numpy(np.ascontiguousarray(img_p)).permute(2, 0, 1).float()
        # Classes 1-24 become 0-23, and class 0 (including padded border) becomes 255 (ignore_index)
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask_p)).long()
        mask_tensor = torch.where(mask_tensor == 0, torch.tensor(255, dtype=torch.long), mask_tensor - 1)

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