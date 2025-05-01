# 5RPF-MAD: Meta Robust SSL Benchmark with t-SNE Visualizations

本项目是一个用于研究自监督学习（SSL）模型鲁棒性和对抗性行为的工具集，包含多种数据处理、模型训练、对抗样本生成以及 t-SNE 可视化的实现。

This repository will contain the code and pretrained models for RPF-MAD.

The full codebase will be publicly released upon paper acceptance.
---

## 功能概述

### 数据处理
- **`attacked_dataset_test.py`** 和 **`attacked_dataset_train_with_clean.py`**：加载对抗样本和干净样本。
- **`create_adversarial_example_datasets.py`**：生成对抗样本。

### 模型训练
- **`train_common_tools.py`**：提供模型训练和评估的通用工具。
- **`attack_common_tools.py`**：实现对抗攻击相关的工具函数。

### t-SNE 可视化
- **`t-SNE_c.py`** 和 **`t-SNE_g.py`**：分别用于 CIFAR-10 和 GTSRB 数据集的 t-SNE 可视化。
- **`t-SNE_3D.py`**：实现 3D t-SNE 可视化。
- **`tsne_visualization_c.py`** 和 **`tsne_visualization_g.py`**：优化的 t-SNE 可视化实现，支持多模型对比。

---

## 快速开始

### 环境依赖
请确保安装以下依赖：
- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- Scikit-learn
- tqdm
- Seaborn
- WandB (可选)

安装依赖：
```bash
pip install -r requirements.txt
```

---

## 项目结构

将此文件保存为 `README.md`，并根据实际需求调整内容。
