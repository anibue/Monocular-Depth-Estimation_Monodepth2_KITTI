# Monodepth2 on KITTI – Technical Report

## Problem Definition

Monocular depth estimation predicts a dense depth map from a single RGB image.  
**Input:** 1 × H × W RGB image.  
**Output:** 1 × H × W depth map (meters) or disparity (inverse depth).

Goal: learn a model that generalizes to unseen driving scenes and produces accurate, scale-consistent depth maps from a single view.

## Data Preparation (KITTI)

- Dataset: **KITTI Raw** driving sequences (supports depth, odometry, 2D/3D detection, stereo).
- Subset: Monocular depth estimation using raw stereo sequences; optionally paired with depth annotations for validation.
- Source URL (example subset): `https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0001_sync.zip`
- License: KITTI research/non-commercial terms (registration required for full set).

### Download & Structure

Recommended project layout:

```
data/
  kitti/
    raw/          # downloaded archives unpacked here
    processed/    # cleaned/resized copies
    splits/       # text files listing relative image paths
```

Use `python src/data_preparation.py --dataset_root ./data/kitti --download --prepare_splits` to:
1. Download archive(s) to `raw/`.
2. Verify checksum/size (simple MD5 placeholder in code).
3. Unpack into `raw/`.
4. Clean images (readability check via OpenCV).
5. Copy to `processed/`.
6. Generate `train_files.txt`, `val_files.txt`, `test_files.txt` in `splits/`.

### Integrity & Cleaning

- MD5/size checks (extend with official checksums when available).
- Corrupted image detection via `cv2.imread`.
- Optional manual count checks per sequence.
- Train/val/test split: default 80/10/10 shuffle; stored as text file lists.
- Preprocessing: resize to config (`img_height`, `img_width`), optional flips/color jitter; normalization to [0,1].

### Metadata

- Each line in split files is a relative path to an image (left camera by default). Right camera inferred by replacing `image_02` → `image_03` when present.
- Extend lists to include multiple drives for full training.

## Model Selection – Monodepth2

Architecture:
- **Encoder–decoder** depth network: ResNet backbone (default ResNet-18) + multi-scale decoder producing disparities at scales {0–3}.
- **Pose network:** predicts relative pose between consecutive frames or stereo pairs.
- **Losses:** photometric reconstruction (SSIM + L1), edge-aware smoothness, auto-masking to ignore static/ambiguous pixels (auto-masking is a TODO in this simplified code).
- **Outputs:** multi-scale disparities; converted to depth via inverse transform with min/max depth clamps.

Why Monodepth2 vs simpler baselines:
- Stronger photometric formulation with auto-masking and per-pixel minimum reprojection improves robustness over earlier monocular methods.
- Multi-scale outputs stabilize training and recover fine details.

### PyTorch Implementation

- Encoder: torchvision ResNet-18/50 (ImageNet pretrain optional).
- Decoder: ELU conv blocks, nearest upsampling, 1-channel disparity per scale.
- PoseNet: ResNet-18 feature trunk + small conv head → 6-DoF pose.
- Losses: SSIM + L1 photometric, edge-aware smoothness. Warping uses simplified stereo-based grid sampling.

### TensorFlow Implementation

- Encoder: Keras backbone (MobileNetV2 placeholder for ResNet-18).
- Decoder: lightweight conv + upsampling per scale; pose head with conv + GAP.
- Losses: simplified photometric + smoothness.
- Differences: TF version approximates the PyTorch architecture; full auto-masking and per-scale min reprojection are TODOs (see code comments).

## Optimization Strategy

- Optimizer: Adam (lr default 1e-4, weight decay optional).
- Scheduler (PyTorch): StepLR with gamma 0.5 every `scheduler_step_size` epochs.
- Batch size: default 4 (adjust for GPU memory).
- Epochs: default 20 (tune for full dataset).
- Regularization: edge-aware smoothness, optional weight decay, gradient clipping (default 1.0).
- Mixed precision: optional (`use_amp`/`use_mixed_precision`) for speed.
- Checkpointing: every epoch to `checkpoints/{backend}`; TensorBoard dirs configured.
- Early stopping: not implemented—add based on validation loss.

### Hyperparameter Tuning & Monitoring

- Monitor: photometric loss, smoothness term, validation photometric loss.
- Metrics (post-eval on ground truth): Abs Rel, Sq Rel, RMSE, RMSE log, δ thresholds.
- Tune: learning rate, scheduler step, min/max depth, augmentation strength, encoder choice.

## Evaluation & Results (template)

Fill after experiments:

| Split | Abs Rel ↓ | Sq Rel ↓ | RMSE ↓ | RMSE log ↓ | δ<1.25 ↑ | Notes |
|-------|-----------|----------|--------|------------|----------|-------|
| Val   |           |          |        |            |          |       |
| Test  |           |          |        |            |          |       |

Evaluation procedure: run inference on validation/test, compare predicted depth vs ground truth depth maps (if available) using standard KITTI depth benchmarks.

## Limitations & Future Work

- Monocular scale ambiguity; requires scale alignment at inference.
- Static scenes and textureless regions remain challenging.
- Simplified warping and missing auto-masking in TF version; PyTorch warping is stereo-style approximation.
- Future work: full temporal photometric warping with camera intrinsics, auto-masking, min reprojection, depth hint supervision, better augmentation, transformer/convnext encoders, and distillation across backends.

