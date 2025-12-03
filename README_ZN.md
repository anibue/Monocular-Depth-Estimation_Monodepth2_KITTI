# Monodepth2 在 KITTI（PyTorch + TensorFlow + MindSpore）

中文说明 | [English](README.md)

基于 KITTI 数据集的单目深度估计，采用多后端架构实现 Monodepth2（PyTorch 为主，提供 TensorFlow/Keras 与 MindSpore 镜像实现）。仓库包含训练/推理脚本、数据准备工具、配置文件、Dockerfile 以及可开源的测试用例。

## 环境说明

- 主要开发环境：Windows 11 + VS Code；训练通常在 Linux（CUDA）上进行。
- Conda 环境：`environment.yml`（PyTorch 2.2.x、TensorFlow 2.12、MindSpore 2.2.10、Python 3.10）。
- Docker：`Dockerfile` 基于支持 CUDA 的 Ubuntu 镜像。

## 快速开始

```bash
conda env create -f environment.yml
conda activate monodepth2-kitti
pip install -e .
```

### 数据集

- 数据集：KITTI 原始数据（单目行车序列）
- 示例下载链接：https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0001_sync.zip （仅示例，请按需替换为完整下载清单）
- 许可：遵循 KITTI 条款（科研/非商业用途）。下载完整数据需在 KITTI 官网注册。

推荐的数据目录结构（准备完成后）：

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

下载与划分（下载示例子集并生成划分文件）：

```bash
python src/data_preparation.py --dataset_root ./data/kitti --download --prepare_splits
```

### 训练

PyTorch：

```bash
python src/train_pytorch.py --config config/config_pytorch.yaml
```

TensorFlow：

```bash
python src/train_tensorflow.py --config config/config_tensorflow.yaml
```

MindSpore：

```bash
python src/train_mindspore.py --config config/config_mindspore.yaml
```

恢复训练：保持 `--config` 指向同一配置文件，并将 `logging.checkpoint_dir` 指向已有的 checkpoint 目录。

### 推理

PyTorch：

```bash
python src/inference_pytorch.py --config config/config_pytorch.yaml \
  --checkpoint checkpoints/pytorch/epoch_1.pth \
  --input_image path/to/image.png \
  --output outputs/pytorch
```

TensorFlow：

```bash
python src/inference_tensorflow.py --config config/config_tensorflow.yaml \
  --checkpoint checkpoints/tensorflow \
  --input_image path/to/image.png \
  --output outputs/tensorflow
```

MindSpore：

```bash
python src/inference_mindspore.py --config config/config_mindspore.yaml \
  --checkpoint checkpoints/mindspore/epoch_1.ckpt \
  --input_image path/to/image.png \
  --output outputs/mindspore
```

### 可视化

深度图 PNG 的保存由 `src/visualization.py` 提供（灰度/伪彩色）。亦可在自定义流水线中导入并对 NumPy 数组调用 `save_depth_map`。

### 测试

```bash
pytest
```

测试使用合成数据，无需下载 KITTI。

## 项目结构

- `config/` —— 不同后端的 YAML 配置（PyTorch / TensorFlow / MindSpore）
- `data/` —— 数据集目录（含占位 `README`）
- `models/` —— Monodepth2 实现（`monodepth2_pytorch.py`、`monodepth2_tf.py`、`monodepth2_mindspore.py`）
- `src/` —— 训练/推理脚本、数据集处理、工具、可视化、数据准备（含 MindSpore 入口）
- `tests/` —— 覆盖数据集、模型、推理的 pytest 用例（未安装 MindSpore 时会跳过其测试）
- `docs/` —— 技术报告 `monodepth2_kitti_technical_report.md`
- `environment.yml` —— Conda 环境规范
- `Dockerfile` —— CUDA 运行时镜像
- `pyproject.toml` —— 包元数据

## 开源注意事项

- 请勿将凭据与 KITTI 原始下载内容纳入版本控制。
- 在论文或项目中请引用 KITTI 与 Monodepth2。
- 欢迎通过 PR 贡献；提交前请先运行 `pytest` 确认通过。
