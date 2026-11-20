import glob
import os
import numpy as np
from PIL import Image

train_img_dir = "/mnt/lustre/scratch/nlsas/home/usc/ci/dgf/FiveBillionPixels/train_16bit_raw"
train_gt_dir = "/mnt/lustre/scratch/nlsas/home/usc/ci/dgf/FiveBillionPixels/train_grayscale_masks"

H, V, B = 6908, 7300, 4

img_paths = sorted(glob.glob(os.path.join(train_img_dir, "*.raw")))[:3]
mask_paths = sorted(glob.glob(os.path.join(train_gt_dir, "*.png")))[:3]

for img_p, mask_p in zip(img_paths, mask_paths):
    raw_data = np.fromfile(img_p, dtype="int32", offset=12)
    img = raw_data.reshape((V, H, B))

    mask = np.array(Image.open(mask_p), dtype=np.int64)
    if mask.shape == (H, V):
        mask = mask.T

    print(f"\n=== {os.path.basename(img_p)} ===")
    print("Mask shape:", mask.shape, "| Unique values:", np.unique(mask))
    vals, counts = np.unique(mask, return_counts=True)
    for v, c in zip(vals, counts):
        pct = 100.0 * c / mask.size
        print(f"  Label {v:3d}: {c:10d} px ({pct:6.3f}%)")

    for b in range(B):
        band = img[:, :, b]
        p1, p50, p99 = np.percentile(band, [1, 50, 99])
        print(
            f"  Band {b}: min={band.min()}, max={band.max()}, "
            f"mean={band.mean():.1f}, std={band.std():.1f}, p1={p1:.1f}, p50={p50:.1f}, p99={p99:.1f}"
        )