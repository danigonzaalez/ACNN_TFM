import os
import json
import glob
import random
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

from utils.torch_utils import compute_band_percentiles


def build_splits(train_dir, train_gt_dir, samples):
    """Reproduces the exact deterministic train/val split used in training scripts."""
    all_train_images = sorted(glob.glob(os.path.join(train_dir, "*.raw")))
    all_train_masks = sorted(glob.glob(os.path.join(train_gt_dir, "*.png")))

    if len(all_train_images) == 0 or len(all_train_masks) == 0:
        raise ValueError("No training images or masks found.")

    combined_train = list(zip(all_train_images, all_train_masks))
    random.seed(42)
    random.shuffle(combined_train)
    all_train_images, all_train_masks = zip(*combined_train)

    total_proportion = samples[0] + samples[1]
    val_ratio = samples[1] / total_proportion if total_proportion > 0 else 0.0
    num_val_images = int(len(all_train_images) * val_ratio)

    if num_val_images > 0:
        val_images = list(all_train_images[:num_val_images])
        val_masks = list(all_train_masks[:num_val_images])
        train_images = list(all_train_images[num_val_images:])
        train_masks = list(all_train_masks[num_val_images:])
    else:
        val_images, val_masks = [], []
        train_images = list(all_train_images)
        train_masks = list(all_train_masks)

    return train_images, train_masks, val_images, val_masks


def extract_split_patches(
    image_paths,
    mask_paths,
    out_dir,
    H,
    V,
    B,
    patch_size,
    p_low,
    p_high,
):
    """Pre-extracts normalized image patches (float16, C x P x P) and remapped masks (uint8, P x P)."""
    os.makedirs(out_dir, exist_ok=True)

    pad_v = (patch_size - (V % patch_size)) % patch_size
    pad_h = (patch_size - (H % patch_size)) % patch_size
    v_pad_total = V + pad_v
    h_pad_total = H + pad_h

    n_rows = v_pad_total // patch_size
    n_cols = h_pad_total // patch_size

    scale = p_high - p_low

    for img_idx, (img_path, mask_path) in enumerate(
        tqdm(list(zip(image_paths, mask_paths)), desc=f"Extracting -> {out_dir}")
    ):
        raw_data = np.fromfile(img_path, dtype="int32", offset=12)
        full_image = raw_data.reshape((V, H, B))

        full_mask = np.array(Image.open(mask_path), dtype=np.int64)
        if full_mask.shape == (H, V):
            full_mask = full_mask.T

        if full_image.shape[:2] != full_mask.shape:
            raise ValueError(
                f"Dimension mismatch: RAW={full_image.shape[:2]}, "
                f"MASK={full_mask.shape}"
            )

        # Percentile scaling clipped to [0.0, 1.0] + band reorder [B, G, R, NIR -> R, G, B, NIR]
        img_float = full_image.astype(np.float32)
        if p_low is not None and p_high is not None:
            img_float = np.clip((img_float - p_low) / scale, 0.0, 1.0)
        else:
            img_float = img_float / 65535.0

        img_norm = img_float[:, :, [2, 1, 0, 3]]

        if pad_v > 0 or pad_h > 0:
            img_norm = np.pad(
                img_norm,
                ((0, pad_v), (0, pad_h), (0, 0)),
                mode="constant",
                constant_values=0.0,
            )
            full_mask = np.pad(
                full_mask,
                ((0, pad_v), (0, pad_h)),
                mode="constant",
                constant_values=0,
            )

        # Remap classes: 1..24 -> 0..23, and 0 (including padded border) -> 255 (ignore_index)
        remapped_mask = np.where(full_mask == 0, 255, full_mask - 1).astype(np.uint8)

        # Channels-first (C, V_pad, H_pad) in float16 to save disk space and I/O bandwidth
        img_chw = np.transpose(img_norm, (2, 0, 1)).astype(np.float16)

        patch_counter = 0
        for row in range(n_rows):
            y0 = row * patch_size
            y1 = y0 + patch_size
            for col in range(n_cols):
                x0 = col * patch_size
                x1 = x0 + patch_size

                img_patch = np.ascontiguousarray(img_chw[:, y0:y1, x0:x1])
                mask_patch = np.ascontiguousarray(remapped_mask[y0:y1, x0:x1])

                base_name = f"img{img_idx:03d}_p{patch_counter:04d}"
                np.save(os.path.join(out_dir, f"{base_name}_img.npy"), img_patch)
                np.save(os.path.join(out_dir, f"{base_name}_mask.npy"), mask_patch)
                patch_counter += 1


def main():
    parser = argparse.ArgumentParser(description="Pre-extract image and mask patches to disk.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.json")
    parser.add_argument(
        "--out-root",
        type=str,
        default=None,
        help="Root directory for pre-extracted patches (defaults to $STORE/patches_<PATCH_SIZE>)",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    train_dir = config.get("train_dir")
    train_gt_dir = config.get("train_gt_dir")
    test_dir = config.get("test_dir")
    test_gt_dir = config.get("test_gt_dir")
    samples = config.get("SAMPLES", [0.9, 0.1])
    patch_size = int(config.get("PATCH_SIZE", 1024))

    store_env = os.environ.get("STORE", "/mnt/netapp2/Store_uni/home/usc/ci/dgf")
    out_root = args.out_root or os.path.join(store_env, f"patches_{patch_size}")

    H, V, B = 6908, 7300, 4

    train_images, train_masks, val_images, val_masks = build_splits(train_dir, train_gt_dir, samples)
    test_images = sorted(glob.glob(os.path.join(test_dir, "*.raw")))
    test_masks = sorted(glob.glob(os.path.join(test_gt_dir, "*.png")))

    if len(test_images) == 0 or len(test_masks) == 0:
        raise ValueError("No test images or masks found.")

    p_low, p_high = compute_band_percentiles(train_images, H=H, V=V, B=B, q_low=1.0, q_high=99.0)
    print(f"* Computed train percentiles p1: {p_low.tolist()} | p99: {p_high.tolist()}")
    print(f"* Saving pre-extracted patches (size={patch_size}) to: {out_root}")

    os.makedirs(out_root, exist_ok=True)
    meta_path = os.path.join(out_root, "metadata.json")
    with open(meta_path, "w") as mf:
        json.dump(
            {
                "H": H,
                "V": V,
                "B": B,
                "patch_size": patch_size,
                "samples": samples,
                "p_low": p_low.tolist(),
                "p_high": p_high.tolist(),
                "num_train_images": len(train_images),
                "num_val_images": len(val_images),
                "num_test_images": len(test_images),
            },
            mf,
            indent=2,
        )

    extract_split_patches(train_images, train_masks, os.path.join(out_root, "train"), H, V, B, patch_size, p_low, p_high)

    if len(val_images) > 0:
        extract_split_patches(val_images, val_masks, os.path.join(out_root, "val"), H, V, B, patch_size, p_low, p_high)
    
    extract_split_patches(test_images, test_masks, os.path.join(out_root, "test"), H, V, B, patch_size, p_low, p_high)

    print("* Pre-extraction completed successfully.")


if __name__ == "__main__":
    main()