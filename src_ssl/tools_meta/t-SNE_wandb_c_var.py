import os
import sys
import time
import wandb
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

sys.path.append(r'/home/users/pxx/workplace/5Adversarial/4Meta_AT_ResNet')
from configs.experiment_config import get_loader, get_model_ACL, load_victim

# 常量定义
NUM_CLASSES = 10

def compute_variances(embeddings, labels):
    """计算类内和类间方差"""
    class_means, class_variances = [], []
    for label in range(NUM_CLASSES):
        mask = labels == label
        if not np.any(mask):
            continue
        class_emb = embeddings[mask]
        class_mean = np.mean(class_emb, axis=0)
        class_means.append(class_mean)
        class_variances.append(np.mean(np.linalg.norm(class_emb - class_mean, axis=1)**2))
    
    intra = np.mean(class_variances)
    overall_mean = np.mean(embeddings, axis=0)
    inter = np.mean([np.linalg.norm(cm - overall_mean)**2 for cm in class_means])
    return intra, inter

def visualize_variance(models, intra_vars, inter_vars):
    """可视化类内和类间方差（柱状图）"""
    fig, ax = plt.subplots(figsize=(10, 5))
    x_labels = models
    x = np.arange(len(models))
    
    ax.bar(x - 0.2, intra_vars, width=0.4, label="Intra-class Variance", color='royalblue')
    ax.bar(x + 0.2, inter_vars, width=0.4, label="Inter-class Variance", color='tomato')

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_ylabel("Variance")
    ax.set_title("Intra-class & Inter-class Variance Comparison")
    ax.legend()
    plt.tight_layout()

    # 保存本地
    plt.savefig("result_tsne/variance_comparison.png")
    plt.close()

    return wandb.Image("result_tsne/variance_comparison.png")

if __name__ == "__main__":
    # 初始化参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIFAR_10')
    parser.add_argument('--data', type=str, default=r'/home/users/pxx/workplace/Datasets/cifar10')
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--resize', type=int, default=32)
    parser.add_argument('--load', type=str, default='FT_ACL', choices=['SSL', 'ACL', 'FT_ACL', 'Meta_ACL'])
    args = parser.parse_args()

    # 获取数据加载器
    train_loader, val_loader, test_loader, num_classes, args = get_loader(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 确定模型列表
    if args.load == "FT_ACL":
        MODEL_CONFIGS = [
            {"name": "ACL", "model_path": "path_to_ACL_model", "load_type": "FT_ACL", "victim": "ACL"},
            {"name": "AdvCL", "model_path": "path_to_AdvCL_model", "load_type": "FT_ACL", "victim": "AdvCL"},
        ]
    else:
        MODEL_CONFIGS = [
            {"name": "Meta_ACL", "model_path": "path_to_Meta_ACL_model", "load_type": "Meta_ACL", "victim": "Meta_ACL"},
            {"name": "Meta_AdvCL", "model_path": "path_to_Meta_AdvCL_model", "load_type": "Meta_ACL", "victim": "Meta_AdvCL"},
        ]

    # 初始化 WandB
    wandb.init(project="tsne_variance_visualization", name=f"{args.load}_{args.dataset}")

    # 结果收集
    results = []
    models, intra_vars, inter_vars = [], [], []

    # 遍历所有模型
    for config in MODEL_CONFIGS:
        model = get_model_ACL(args, device)
        model.eval()
        embeddings, labels = [], []
        
        with torch.no_grad():
            for images, targets in tqdm(test_loader):
                emb = model(images.to(device), return_features=True)
                embeddings.append(emb.cpu().numpy())
                labels.append(targets.numpy())

        embeddings = np.concatenate(embeddings)
        labels = np.concatenate(labels)

        # 计算类内和类间方差
        intra, inter = compute_variances(embeddings, labels)
        models.append(config["name"])
        intra_vars.append(intra)
        inter_vars.append(inter)

        results.append(f"{config['name']:<15} Intra: {intra:.2f} Inter: {inter:.2f}")

    # 保存方差计算结果
    with open("result_tsne/variance_results.txt", 'w') as f:
        f.write("\n".join(results))

    # 可视化方差对比
    variance_img = visualize_variance(models, intra_vars, inter_vars)

    # 上传到 WandB
    wandb.log({"Variance Comparison": variance_img})
    wandb.finish()

    print("类内/类间方差可视化完成，查看 WandB 结果！")
