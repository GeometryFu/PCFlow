<div align="center">

# PCFlow

## Physics-Conditioned Flow Matching for GPR Pipeline Synthesis

[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![Language](https://img.shields.io/badge/Lang-English-blue.svg)](README.md) [![Language](https://img.shields.io/badge/Lang-中文-red.svg)](README_CN.md)

🔥 **PCFlow** 是一个用于探地雷达（GPR）B-scan 快速合成的框架，基于麦克斯韦物理条件场引导的流匹配实现，实现了高视觉保真度与强物理一致性的统一。

### ⚡ 传统数值仿真：分钟 / 小时级 → PCFlow：秒级

<p float="center">
  <img src="assets/pipeline.png" width="90%" />
</p>

</div>

## 📋 目录

- [🚀 快速开始](#-快速开始)
- [📦 环境安装](#-环境安装)
- [🧠 推理](#-推理)
- [🏋️ 训练](#️-训练)

## 🚀 快速开始

1️⃣ 克隆仓库并进入可执行源码目录

```bash
git clone https://github.com/GeometryFu/PCFlow.git
cd PCFlow
```

2️⃣ 创建环境并安装依赖

```bash
conda create -n pcflow python=3.10 -y
conda activate pcflow

# 请先安装与本机 CUDA 匹配的 PyTorch
pip install -r requirements.txt
```

3️⃣ 使用内置数据集开始训练

```bash
python train.py \
    --model-config configs/model_gpr.yaml \
    --train-config configs/train_gpr.yaml
```

4️⃣ 使用训练检查点运行推理 🎉

```bash
python sample.py \
    --ckpt outputs/gpr_cfm/checkpoints/ckpt_0050000.pt \
    --image dataset_split/images/test_id/steel_free_space_r0.08_eps10_sig0.005_x2.29_y0.65.png \
    --infile dataset_split/conditions/test_id/steel_free_space_r0.08_eps10_sig0.005_x2.29_y0.65.in \
    --out outputs/sample.png
```

📷 训练输出示例

```text
Resume disabled; training from scratch.
Step=50 | Total=...
[VAL] Step=1000 | ValLoss=... | ValMSE=...
Saved checkpoint at step 1000
```

## 📦 环境安装

### 依赖项

- Python >= 3.10
- PyTorch >= 2.1
- 推荐使用支持 CUDA 的 GPU

### 安装

请先从 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 选择与本机 CUDA 环境匹配的版本，再安装项目依赖：

```bash
cd PCFlow
pip install -r requirements.txt
```

默认从 `stabilityai/sdxl-vae` 加载冻结的 VAE。离线使用时，请提前下载该模型，并将 `configs/model_gpr.yaml` 中的 `vae.pretrained_path` 改为本地目录。

### 验证安装

```bash
python train.py --help
python sample.py --help
```

## 🧠 推理

```bash
# 使用显式指定的图片 / gprMax .in 配对文件生成
python sample.py \
    --model-config configs/model_gpr.yaml \
    --train-config configs/train_gpr.yaml \
    --ckpt outputs/gpr_cfm/checkpoints/ckpt_0050000.pt \
    --image path/to/reference.png \
    --infile path/to/scene.in \
    --out outputs/sample.png

# 或使用 JSONL 索引中的第一条配对记录
python sample.py \
    --ckpt outputs/gpr_cfm/checkpoints/ckpt_0050000.pt \
    --jsonl dataset_split/jsonl/test_id.jsonl \
    --out outputs/sample.png
```

采样脚本根据场景重建物理条件，优先使用 EMA 权重，按照配置选择 Euler 或 Heun 求解器积分整流流 ODE，并通过 SDXL VAE 解码结果。

### 完整参数说明

| 参数 | 是否必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--model-config` | 否 | `configs/model_gpr.yaml` | 模型与物理编码器配置 |
| `--train-config` | 否 | `configs/train_gpr.yaml` | 设备、求解器与采样配置 |
| `--ckpt` | 是 | — | 训练得到的 `.pt` 检查点 |
| `--jsonl` | 否 | — | JSONL 索引；当前脚本读取第一条记录 |
| `--image` | 与 `--infile` 同时使用 | — | 当前接口要求的配对参考 B-scan 路径 |
| `--infile` | 与 `--image` 同时使用 | — | gprMax 风格物理场景文件 |
| `--out` | 是 | — | 输出图片路径 |

## 🏋️ 训练

```bash
# 主单进程训练配置
python train.py

# 选择备选数据划分配置
python train.py \
    --model-config configs/model_gpr.yaml \
    --train-config configs/train_gpr_dataset_split.yaml
```

训练由两个 YAML 文件控制：

- `configs/model_gpr.yaml`：冻结的 SDXL VAE、26 通道 Maxwell 条件编码器和条件 U-Net 架构。
- `configs/train_gpr.yaml`：数据路径、优化器、训练计划、EMA、验证、检查点与 ODE 求解器配置。

训练循环包含以下步骤：

1. 将每个 gprMax `.in` 文件解析为介电常数、电导率和几何图。
2. 构建 26 通道物理条件，并降采样到潜空间分辨率。
3. 将灰度 B-scan 转为伪 RGB，通过 SDXL VAE 编码为潜变量 `z1`。
4. 采样高斯噪声 `z0` 和 `t ~ U(0,1)`，构造 `zt = (1-t)z0 + tz1`。
5. 使用深度加权 MSE 训练 U-Net 预测 `v = z1 - z0`。主配置同时使用 10% 条件丢弃、AMP、梯度裁剪和 EMA。

主配置训练 50,000 步，batch size 为 8，AdamW 学习率为 `5e-5`；每 50 步记录日志，每 500 步生成固定验证样本，每 1,000 步验证并保存检查点。请根据硬件调整 `batch_size` 和 `num_workers`。当前入口为单进程程序，使用 YAML 中配置的设备。

### 恢复训练

断点恢复通过训练 YAML 设置，而不是命令行 `--resume` 参数：

```yaml
train:
  resume:
    enabled: true
    checkpoint_path: "ckpt_0010000.pt"  # null 表示自动选择最新检查点
```

当前实现会恢复 U-Net、EMA 权重和训练步数。

### 数据集准备

内置 800 组场景数据已经按如下结构建立索引：

```text
dataset_split/
├── images/          # train / val / test_id / stress_test B-scan PNG
├── conditions/      # 配对的 gprMax .in 文件
├── jsonl/           # 数据加载器使用的索引
│   └── fewshot/     # 50 / 100 / 200 / 300 / full-443 子集
├── metadata/        # 各划分元数据表
├── summary.json     # 数据集统计
└── LICENSE          # CC BY 4.0 数据集许可证
```

自定义配对数据的图片和 `.in` 文件必须同名（扩展名除外）：

```bash
python data/gpr_dataset/build_index.py \
    --images_dir path/to/images \
    --ins_dir path/to/conditions \
    --out_jsonl path/to/train.jsonl
```

随后在所选训练 YAML 中修改 `data.train_jsonl`、`data.val_jsonl` 等路径。训练输出写入 `output.workdir`，其中包含检查点、固定样本、CSV 日志与损失曲线。

🏠 **Project Page**: [https://GeometryFu.github.io/PCFlow](https://GeometryFu.github.io/PCFlow)

<div align="center">

<img src="assets/mascot.png" width="120" alt="PCFlow Mascot"/>

**PCFlow 代码**采用 [Apache 2.0 License](LICENSE)。<br>
内置数据集采用 [CC BY 4.0](dataset_split/LICENSE)。

Made with ❤️ by the PCFlow Team

</div>
