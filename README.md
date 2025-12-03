# Monodepth2 on KITTI (PyTorch + TensorFlow + MindSpore)

Monocular depth estimation with a dual-backend Monodepth2 implementation targeting the KITTI dataset. This repo includes a PyTorch-first training/inference pipeline, a TensorFlow/Keras mirror, data preparation scripts, configs, Dockerfile, and tests suitable for open-sourcing.

## Environment

- Primary dev: Windows 11 + VS Code; training typically on Linux (CUDA).
- Conda environment: `environment.yml` (PyTorch 2.2.x, TensorFlow 2.12, MindSpore 2.2.10, Python 3.10).
- Docker: CUDA-enabled base image (Ubuntu) in `Dockerfile`

## Quickstart

```bash
conda env create -f environment.yml
conda activate monodepth2-kitti
pip install -e .
```

### Data

- Dataset: KITTI raw data (monocular driving sequences)
- Source URL: https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0001_sync.zip (example subset; replace with your full download list)
- License: KITTI terms (research/non-commercial). Register on KITTI for full downloads.

Recommended layout after preparation:

```
data/
  kitti/
    raw/
    processed/
    splits/
      train_files.txt
      val_files.txt
      test_files.txt
```

Download & split:

```bash
python src/data_preparation.py --dataset_root ./data/kitti --download --prepare_splits
```

### Training

PyTorch:

```bash
python src/train_pytorch.py --config config/config_pytorch.yaml
```

TensorFlow:

```bash
python src/train_tensorflow.py --config config/config_tensorflow.yaml
```

MindSpore:

```bash
python src/train_mindspore.py --config config/config_mindspore.yaml
```

Resume training: point `--config` to same file and set `logging.checkpoint_dir` to existing checkpoints.

### Inference

PyTorch:

```bash
python src/inference_pytorch.py --config config/config_pytorch.yaml \
  --checkpoint checkpoints/pytorch/epoch_1.pth \
  --input_image path/to/image.png \
  --output outputs/pytorch
```

TensorFlow:

```bash
python src/inference_tensorflow.py --config config/config_tensorflow.yaml \
  --checkpoint checkpoints/tensorflow \
  --input_image path/to/image.png \
  --output outputs/tensorflow
```

MindSpore:

```bash
python src/inference_mindspore.py --config config/config_mindspore.yaml \
  --checkpoint checkpoints/mindspore/epoch_1.ckpt \
  --input_image path/to/image.png \
  --output outputs/mindspore
```

### Visualization

Depth PNGs are saved via `src/visualization.py` (grayscale/colormap). You can also import and call `save_depth_map` on NumPy arrays for custom pipelines.

### Tests

```bash
pytest
```

Tests use synthetic data; no KITTI download needed.

## Project Structure

- `config/` – YAML configs for PyTorch, TensorFlow, and MindSpore
- `data/` – dataset folder (placeholder README inside)
- `models/` – Monodepth2 implementations (`monodepth2_pytorch.py`, `monodepth2_tf.py`, `monodepth2_mindspore.py`)
- `src/` – training, inference, datasets, utils, visualization, data prep (plus MindSpore entrypoints)
- `tests/` – pytest suites for datasets, models, and inference (MindSpore tests skipped if not installed)
- `docs/` – `monodepth2_kitti_technical_report.md`
- `environment.yml` – conda env spec
- `Dockerfile` – CUDA-enabled runtime image
- `pyproject.toml` – package metadata

## Open-Source Notes

- Keep credentials and KITTI downloads out of version control.
- Cite KITTI and Monodepth2 in papers/projects.
- Contributions welcome via PRs; please run `pytest` before submitting.
