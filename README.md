
<div align="center">

<!-- <img src="assets/mascot.png" width="200" alt="PCFlow Mascot"/> -->

# PCFlow

**PCFlow: Physics-Conditioned Flow Matching for GPR Pipeline Synthesis**

[![arXiv](https://img.shields.io/badge/arXiv-2501.xxxxx-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2501.xxxxx)
[![HuggingFace](https://img.shields.io/badge/🤗_Weights-Coming_Soon-yellow.svg)](https://huggingface.co/xxx/PCFlow)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/GeometryFu/PCFlow?style=social)](https://github.com/GeometryFu/PCFlow)

[//]: # "上方 badge 替换你的真实链接，HuggingFace 暂时保留 Coming_Soon 即可"

🔥 **PCFlow** 是一个用于 XXX 的框架，基于 XX 实现，在 XX 任务上达到 SOTA。


<p float="center">
  <img src="assets/pipeline.png" width="90%" />
</p>

</div>

---

## 📋 目录

- [🚀 快速开始](#-快速开始)
- [📦 环境安装](#-环境安装)
- [🧠 推理](#-推理)
- [🏋️ 训练](#️-训练)
- [📊 结果](#-结果)
- [📖 论文](#-论文)


## 🚀 快速开始

1️⃣ 克隆仓库
```bash
git clone https://github.com/GeometryFu/PCFlow.git
cd PCFlow
```

2️⃣ 创建环境 & 安装依赖
```bash
conda create -n pcflow python=3.10 -y
conda activate pcflow
pip install -r requirements.txt
```

3️⃣ 下载预训练权重（HuggingFace）
```bash
mkdir -p checkpoints
# wget -P checkpoints https://huggingface.co/xxx/PCFlow/resolve/main/pcflow_base.pth
```

4️⃣ 运行推理 🎉
```bash
python inference.py \
    --config configs/pcflow_base.yaml \
    --checkpoint checkpoints/pcflow_base.pth \
    --input examples/sample_input.png \
    --output results/
```


📷 预期输出

```
✅ Loading checkpoint from checkpoints/pcflow_base.pth
✅ Model loaded successfully | Params: 85.2M
✅ Running inference on examples/sample_input.png
✅ Result saved to results/sample_output.png
⏱  Inference time: 0.42s (single A100)
```

## 📦 环境安装


### 依赖项

- Python >= 3.10
- PyTorch >= 2.1
- CUDA >= 11.8（推荐）

### 安装

```bash
# 方式一：pip
pip install -r requirements.txt

# 方式二：editable 安装（开发推荐）
pip install -e .
```

### 验证安装

```bash
python -c "import pcflow; print(pcflow.__version__)"
# 输出: 0.1.0
```

---

## 🧠 推理

```bash
# 单张推理
python inference.py \
    --config configs/pcflow_base.yaml \
    --checkpoint checkpoints/pcflow_base.pth \
    --input path/to/your/input \
    --output results/

# 批量推理
python inference.py \
    --config configs/pcflow_base.yaml \
    --checkpoint checkpoints/pcflow_base.pth \
    --input_dir path/to/input_folder \
    --output results/

# 使用不同模型规格
python inference.py \
    --config configs/pcflow_large.yaml \
    --checkpoint checkpoints/pcflow_large.pth \
    --input path/to/your/input \
    --output results/
```

## ⚙️ 完整参数说明

| 参数            | 类型 | 默认值                     | 说明           |
| --------------- | ---- | -------------------------- | -------------- |
| `--config`      | str  | `configs/pcflow_base.yaml` | 配置文件路径   |
| `--checkpoint`  | str  | 必填                       | 模型权重路径   |
| `--input`       | str  | None                       | 单张输入路径   |
| `--input_dir`   | str  | None                       | 批量输入文件夹 |
| `--output`      | str  | `results/`                 | 输出目录       |
| `--device`      | str  | `cuda`                     | 推理设备       |
| `--seed`        | int  | `42`                       | 随机种子       |
| `--num_samples` | int  | `1`                        | 采样数量       |

</details>

---

## 🏋️ 训练

```bash
# 单卡训练
python train.py --config configs/pcflow_base.yaml

# 多卡训练（推荐）
torchrun --nproc_per_node=4 train.py --config configs/pcflow_base.yaml

# 恢复训练
python train.py --config configs/pcflow_base.yaml --resume checkpoints/pcflow_base_epoch10.pth
```

## 📁 数据集准备

请将数据集组织为如下结构：

```
data/
├── dataset_name/
│   ├── train/
│   │   ├── 00001.png
│   │   ├── 00002.png
│   │   └── ...
│   ├── val/
│   │   └── ...
│   └── metadata.csv
```

修改 `configs/pcflow_base.yaml` 中的 `data.root` 字段指向你的数据路径。


---

## 📖 论文

### PCFlow: Physics-Conditioned Flow Matching for GPR Pipeline Synthesis

📄 **Paper**: [arXiv Link (Coming Soon)](https://arxiv.org/abs/2501.xxxxx)

🏠 **Project Page**: [https://GeometryFu.github.io/PCFlow](https://GeometryFu.github.io/PCFlow)

🤗 **Model Weights**: [HuggingFace (Coming Soon)](https://huggingface.co/xxx/PCFlow)

---

<div align="center">

<img src="assets/mascot.png" width="120" alt="PCFlow Mascot"/>

**PCFlow** is released under the [Apache 2.0 License](LICENSE).

Made with ❤️ by the PCFlow Team

</div>
