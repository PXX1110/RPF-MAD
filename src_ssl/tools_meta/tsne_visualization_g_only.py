import os
import sys
import time
sys.path.append(r'/home/users/pxx/workplace/5Adversarial/4Meta_AT_ResNet')
import torch
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from configs.experiment_config import get_loader, get_model_ACL, load_victim
import seaborn as sns

# 模型配置列表（根据实际路径修改）
MODEL_CONFIGS_FT_ACL = [
    {   # ACL 配置
        "name": "ACL",
        "model_path": r"/home/users/pxx/workplace/5Adversarial/9_2Robust_SSL/RobustSSL_Benchmark-main/checkpoints/ACL_cifar10_r18_gtsrb/AFF/AFF_model_bestAT.pt",
        "load_type": "FT_ACL",
        "victim": "ACL"
    },
    {   # AdvCL 配置
        "name": "AdvCL",
        "model_path": r"/home/users/pxx/workplace/5Adversarial/9_2Robust_SSL/RobustSSL_Benchmark-main/checkpoints/AdvCL_cifar10_r18_gtsrb/AFF/AFF_model_bestAT.pt",
        "load_type": "FT_ACL",
        "victim": "AdvCL"
    },
    {   # A-InfoNCE 配置
        "name": "A-InfoNCE",
        "model_path": r"/home/users/pxx/workplace/5Adversarial/9_2Robust_SSL/RobustSSL_Benchmark-main/checkpoints/AInfoNCE_cifar10_r18_gtsrb/AFF/AFF_model_bestAT.pt",
        "load_type": "FT_ACL",
        "victim": "A-InfoNCE"
    },
    {   # DynACL 配置
        "name": "DynACL",
        "model_path": r"/home/users/pxx/workplace/5Adversarial/9_2Robust_SSL/RobustSSL_Benchmark-main/checkpoints/DynACL_cifar10_r18_gtsrb/AFF/AFF_model_bestAT.pt",
        "load_type": "FT_ACL",
        "victim": "DynACL"
    },
    {   # DynACL++ 配置
        "name": "DynACL++",
        "model_path": r"/home/users/pxx/workplace/5Adversarial/9_2Robust_SSL/RobustSSL_Benchmark-main/checkpoints/DynACL++_cifar10_r18_gtsrb/AFF/AFF_model_bestAT.pt",
        "load_type": "FT_ACL",
        "victim": "DynACL++"
    },
    {   # DynACL-AIR 配置
        "name": "DynACL-AIR",
        "model_path": r"/home/users/pxx/workplace/5Adversarial/9_2Robust_SSL/RobustSSL_Benchmark-main/checkpoints/DynACL_AIR_cifar10_r18_gtsrb/AFF/AFF_model_bestAT.pt",
        "load_type": "FT_ACL",
        "victim": "DynACL-AIR"
    },
    {   # DynACL-AIR++ 配置
        "name": "DynACL-AIR++",
        "model_path": r"/home/users/pxx/workplace/5Adversarial/9_2Robust_SSL/RobustSSL_Benchmark-main/checkpoints/DynACL_AIR++_cifar10_r18_gtsrb/AFF/AFF_model_bestAT.pt",
        "load_type": "FT_ACL",
        "victim": "DynACL-AIR++"
    },
    {   # DynACL-RCS 配置
        "name": "DynACL-RCS",
        "model_path": r"/home/users/pxx/workplace/5Adversarial/9_2Robust_SSL/RobustSSL_Benchmark-main/checkpoints/DynACL_RCS_cifar10_r18_gtsrb/AFF/AFF_model_bestAT.pt",
        "load_type": "FT_ACL",
        "victim": "DynACL-RCS"
    }
]

MODEL_CONFIGS_Meta_ACL = [
    {   # ACL 配置
        "name": "ACL",
        "model_path": r"/home/users/pxx/workplace/5Adversarial/9_2Robust_SSL/RobustSSL_Benchmark-main/checkpoints/SSL_cifar10_r18_gtsrb_meta_0.01_cross_LAMBDA3_1/AFF/AFF_model_bestAT.pt",
        "load_type": "Meta_ACL",
        "victim": "Meta_ACL"
    },
    {   # AdvCL 配置
        "name": "AdvCL",
        "model_path": r"/home/users/pxx/workplace/5Adversarial/9_2Robust_SSL/RobustSSL_Benchmark-main/checkpoints/SSL_cifar10_r18_gtsrb_Meta_AdvCL_all_c/AFF/AFF_model_bestAT.pt",
        "load_type": "Meta_ACL",
        "victim": "Meta_AdvCL"
    },
    {   # A-InfoNCE 配置
        "name": "A-InfoNCE",
        "model_path": r"/home/users/pxx/workplace/5Adversarial/9_2Robust_SSL/RobustSSL_Benchmark-main/checkpoints/SSL_cifar10_r18_gtsrb_Meta_A-InfoNCE_all_c/AFF/AFF_model_bestAT.pt",
        "load_type": "Meta_ACL",
        "victim": "Meta_A-InfoNCE"
    },
    {   # DynACL 配置
        "name": "DynACL",
        "model_path": r"/home/users/pxx/workplace/5Adversarial/9_2Robust_SSL/RobustSSL_Benchmark-main/checkpoints/SSL_cifar10_r18_gtsrb_Meta_DynACL/AFF/AFF_model_bestAT.pt",
        "load_type": "Meta_ACL",
        "victim": "Meta_DynACL"
    },
    {   # DynACL++ 配置
        "name": "DynACL++",
        "model_path": r"/home/users/pxx/workplace/5Adversarial/9_2Robust_SSL/RobustSSL_Benchmark-main/checkpoints/SSL_cifar10_r18_gtsrb_Meta_DynACL++/AFF/AFF_model_bestAT.pt",
        "load_type": "Meta_ACL",
        "victim": "Meta_DynACL++"
    },
    {   # DynACL-AIR 配置
        "name": "DynACL-AIR",
        "model_path": r"/home/users/pxx/workplace/5Adversarial/9_2Robust_SSL/RobustSSL_Benchmark-main/checkpoints/SSL_cifar10_r18_gtsrb_Meta_DynACL-AIR/AFF/AFF_model_bestAT.pt",
        "load_type": "Meta_ACL",
        "victim": "Meta_DynACL-AIR"
    },
    {   # DynACL-AIR++ 配置
        "name": "DynACL-AIR++",
        "model_path": r"/home/users/pxx/workplace/5Adversarial/9_2Robust_SSL/RobustSSL_Benchmark-main/checkpoints/SSL_cifar10_r18_gtsrb_Meta_DynACL-AIR++/AFF/AFF_model_bestAT.pt",
        "load_type": "Meta_ACL",
        "victim": "Meta_DynACL-AIR++"
    },
    {   # DynACL-RCS 配置
        "name": "DynACL-RCS",
        "model_path": r"/home/users/pxx/workplace/5Adversarial/9_2Robust_SSL/RobustSSL_Benchmark-main/checkpoints/SSL_cifar10_r18_gtsrb_Meta_DynACL-RCS/AFF/AFF_model_bestAT.pt",
        "load_type": "Meta_ACL",
        "victim": "Meta_DynACL-RCS"
    }
]

# 设置 Seaborn 配色风格
sns.set(style="white", font_scale=1.2)

# **加载模型**
def load_model(config, args, num_classes, device):
    if config["load_type"] == "SSL":
        args.victim = config["victim"]
        model = load_victim(args, device)
    elif config["load_type"] == "ACL":
        model = get_model_ACL(args, device)
    elif config["load_type"] in ["FT_ACL", "Meta_ACL"]:
        from model.resnet_ssl import resnet18
        args.victim = config["victim"]
        do_normalize = 0 if args.victim in ['AdvCL', 'A-InfoNCE', 'Meta_AdvCL', 'Meta_A-InfoNCE'] else 1
        model = resnet18(num_classes=num_classes, do_normalize=do_normalize)
        ckpt = torch.load(config["model_path"])
        model.load_state_dict(ckpt['state_dict'])
        model.to(device)
    model.eval()
    return model

# **特征提取**
def extract_embeddings(model, loader, device):
    embeddings, labels = [], []
    with torch.no_grad():
        for images, targets in tqdm(loader):
            emb = model(images.to(device), return_features=True)
            embeddings.append(emb.cpu().numpy())
            labels.append(targets.numpy())
    return np.concatenate(embeddings), np.concatenate(labels)

# **t-SNE 可视化**
def visualize_tsne(embeddings, labels, ax, title):
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
    points = tsne.fit_transform(embeddings)

    # 获取颜色映射（`tab10` 使分类清晰）
    num_classes = len(set(labels))
    palette = sns.color_palette("tab10", num_classes)
    
    # 绘制散点图
    scatter = ax.scatter(points[:, 0], points[:, 1], c=[palette[i] for i in labels], s=3, alpha=0.3)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(title, fontsize=10, labelpad=6)  # **标题放在下方**
    ax.set_frame_on(False)  # 去掉边框
    return scatter

if __name__ == "__main__":
    # 初始化参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='gtsrb')
    parser.add_argument('--data', type=str, default=r'/home/users/pxx/workplace/Datasets/GTSRB')
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--resize', type=int, default=32, help='location of the data')
    parser.add_argument('--load', type=str, default='Meta_ACL', choices=['SSL', 'ACL', 'FT_ACL', 'Meta_ACL'])
    args = parser.parse_args()

    PLOT_SAVE_PATH_EPS = os.path.join("result_tsne", f"comparative_tsne_GTSRB_{args.load}.eps")
    PLOT_SAVE_PATH_PNG = os.path.join("result_tsne", f"comparative_tsne_GTSRB_{args.load}.png")

    # 获取数据加载器
    train_loader, val_loader, test_loader, num_classes, args = get_loader(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # **创建 4×2 可视化画布**
    fig, axs = plt.subplots(2, 4, figsize=(14, 7))

    if args.load == "FT_ACL":
        MODEL_CONFIGS = MODEL_CONFIGS_FT_ACL
    else:
        MODEL_CONFIGS = MODEL_CONFIGS_Meta_ACL

    for idx, config in enumerate(MODEL_CONFIGS):
        model = load_model(config, args, num_classes, device)
        embeddings, labels = extract_embeddings(model, test_loader, device)
        
        row, col = divmod(idx, 4)  # 4 行 2 列
        visualize_tsne(embeddings, labels, axs[row, col], config['name'])

    # **调整布局**
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 保存为 EPS 格式
    plt.savefig(PLOT_SAVE_PATH_EPS, dpi=300, format='eps', bbox_inches='tight')
    # 保存为 PNG 格式
    plt.savefig(PLOT_SAVE_PATH_PNG, dpi=300, format='png', bbox_inches='tight')
    # 关闭窗口
    plt.close()
    time.sleep(10)