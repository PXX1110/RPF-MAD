import os
import sys
import time
sys.path.append(r'/home/users/pxx/workplace/5Adversarial/4Meta_AT_ResNet')
import torch
import argparse
from tqdm import tqdm
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
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

# 模型配置列表（根据实际路径修改）
MODEL_CONFIGS_FT_ACL = [
    {   # ACL 配置
        "name": "ACL",
        "model_path": r"/home/users/pxx/workplace/5Adversarial/9_2Robust_SSL/RobustSSL_Benchmark-main/checkpoints/SSL_cifar10_r18_cifar10_ACL/AFF/AFF_model_bestAT.pt",
        "load_type": "FT_ACL",
        "victim": "ACL"
    },
    {   # AdvCL 配置
        "name": "AdvCL",
        "model_path": r"/home/users/pxx/workplace/5Adversarial/9_2Robust_SSL/RobustSSL_Benchmark-main/checkpoints/SSL_cifar10_r18_cifar10_AdvCL/AFF/AFF_model_bestAT.pt",
        "load_type": "FT_ACL",
        "victim": "AdvCL"
    },
    {   # A-InfoNCE 配置
        "name": "A-InfoNCE",
        "model_path": r"/home/users/pxx/workplace/5Adversarial/9_2Robust_SSL/RobustSSL_Benchmark-main/checkpoints/SSL_cifar10_r18_cifar10_AInfoNCE/AFF/AFF_model_bestAT.pt",
        "load_type": "FT_ACL",
        "victim": "A-InfoNCE"
    },
    {   # DynACL 配置
        "name": "DynACL",
        "model_path": r"/home/users/pxx/workplace/5Adversarial/9_2Robust_SSL/RobustSSL_Benchmark-main/checkpoints/SSL_cifar10_r18_cifar10_DynACL/AFF/AFF_model_bestAT.pt",
        "load_type": "FT_ACL",
        "victim": "DynACL"
    },
    {   # DynACL++ 配置
        "name": "DynACL++",
        "model_path": r"/home/users/pxx/workplace/5Adversarial/9_2Robust_SSL/RobustSSL_Benchmark-main/checkpoints/SSL_cifar10_r18_cifar10_DynACL++/AFF/AFF_model_bestAT.pt",
        "load_type": "FT_ACL",
        "victim": "DynACL++"
    },
    {   # DynACL-AIR 配置
        "name": "DynACL-AIR",
        "model_path": r"/home/users/pxx/workplace/5Adversarial/9_2Robust_SSL/RobustSSL_Benchmark-main/checkpoints/SSL_cifar10_r18_cifar10_DynACL-AIR/AFF/AFF_model_bestAT.pt",
        "load_type": "FT_ACL",
        "victim": "DynACL-AIR"
    },
    {   # DynACL-AIR++ 配置
        "name": "DynACL-AIR++",
        "model_path": r"/home/users/pxx/workplace/5Adversarial/9_2Robust_SSL/RobustSSL_Benchmark-main/checkpoints/SSL_cifar10_r18_cifar10_DynACL-AIR++/AFF/AFF_model_bestAT.pt",
        "load_type": "FT_ACL",
        "victim": "DynACL-AIR++"
    },
    {   # DynACL-RCS 配置
        "name": "DynACL-RCS",
        "model_path": r"/home/users/pxx/workplace/5Adversarial/9_2Robust_SSL/RobustSSL_Benchmark-main/checkpoints/SSL_cifar10_r18_cifar10_DynACL-RCS/AFF/AFF_model_bestAT.pt",
        "load_type": "FT_ACL",
        "victim": "DynACL-RCS"
    }
]

MODEL_CONFIGS_Meta_ACL = [
    {   # ACL 配置
        "name": "ACL",
        "model_path": r"/home/users/pxx/workplace/5Adversarial/4Meta_AT_ResNet/results_ssl/ACL_SLF_Multi_MAD_TRAIN_earlystop_val_True_lr_max_0.01_0.001_SGD_82_28/best_checkpint.pkl",
        "load_type": "Meta_ACL",
        "victim": "Meta_ACL"
    },
    {   # AdvCL 配置
        "name": "AdvCL",
        "model_path": r"/home/users/pxx/workplace/5Adversarial/4Meta_AT_ResNet/results_ssl/CIFAR_10_Meta_Multi_AT_train_0.001_0.0001_SGD_AdvCL_80_37/best_checkpint.pkl",
        "load_type": "Meta_ACL",
        "victim": "Meta_AdvCL"
    },
    {   # A-InfoNCE 配置
        "name": "A-InfoNCE",
        "model_path": r"/home/users/pxx/workplace/5Adversarial/4Meta_AT_ResNet/results_ssl/CIFAR_10_Meta_Multi_AT_train_0.001_0.0001_SGD_AInfoNCE_83_33/best_checkpint.pkl",
        "load_type": "Meta_ACL",
        "victim": "Meta_A-InfoNCE"
    },
    {   # DynACL 配置
        "name": "DynACL",
        "model_path": r"/home/users/pxx/workplace/5Adversarial/4Meta_AT_ResNet/results_ssl/CIFAR_10_Meta_Multi_AT_train_0.01_0.001_SGD_DynACL_86_34/best_checkpint.pkl",
        "load_type": "Meta_ACL",
        "victim": "Meta_DynACL"
    },
    {   # DynACL++ 配置
        "name": "DynACL++",
        "model_path": r"/home/users/pxx/workplace/5Adversarial/4Meta_AT_ResNet/results_ssl/CIFAR_10_Meta_Multi_AT_train_0.01_0.001_SGD_DynACL++_84_30/best_checkpint.pkl",
        "load_type": "Meta_ACL",
        "victim": "Meta_DynACL++"
    },
    {   # DynACL-AIR 配置
        "name": "DynACL-AIR",
        "model_path": r"/home/users/pxx/workplace/5Adversarial/4Meta_AT_ResNet/results_ssl/CIFAR_10_Meta_Multi_AT_train_0.01_0.001_SGD_DynACL-AIR_78_34/best_checkpint.pkl",
        "load_type": "Meta_ACL",
        "victim": "Meta_DynACL-AIR"
    },
    {   # DynACL-AIR++ 配置
        "name": "DynACL-AIR++",
        "model_path": r"/home/users/pxx/workplace/5Adversarial/4Meta_AT_ResNet/results_ssl/CIFAR_10_Meta_Multi_AT_train_0.01_0.001_SGD_DynACL-AIR++_85_34/best_checkpint.pkl",
        "load_type": "Meta_ACL",
        "victim": "Meta_DynACL-AIR++"
    },
    {   # DynACL-RCS 配置
        "name": "DynACL-RCS",
        "model_path": r"/home/users/pxx/workplace/5Adversarial/4Meta_AT_ResNet/results_ssl/CIFAR_10_Meta_Multi_AT_train_0.01_0.001_SGD_DynACL-RCS_82_29/best_checkpint.pkl",
        "load_type": "Meta_ACL",
        "victim": "Meta_DynACL-RCS"
    }
]

def load_model(config, args, num_classes, device):
    """加载指定配置的模型"""
    if config["load_type"] == "SSL":
        args.victim = config["victim"]
        model = load_victim(args, device)
    elif config["load_type"] == "ACL":
        model = get_model_ACL(args, device)
    elif config["load_type"] in ["FT_ACL", "Meta_ACL"]:
        from model.resnet_ssl import resnet18
        args.victim = config["victim"]
        do_normalize = 0 if args.victim in ['AdvCL', 'A-InfoNCE','Meta_AdvCL', 'Meta_A-InfoNCE'] else 1
        model = resnet18(num_classes=num_classes, do_normalize=do_normalize)
        ckpt = torch.load(config["model_path"])
        model.load_state_dict(ckpt['state_dict'])
        model.to(device)
    model.eval()

    return model

def extract_embeddings(model, loader, device):
    """特征提取优化实现"""
    embeddings, labels = [], []
    with torch.no_grad():
        for images, targets in tqdm(loader):
            emb = model(images.to(device), return_features=True)
            embeddings.append(emb.cpu().numpy())
            labels.append(targets.numpy())
    return np.concatenate(embeddings), np.concatenate(labels)

def visualize_tsne(embeddings, labels, ax, title):
    """可视化优化实现"""
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000)
    points = tsne.fit_transform(embeddings)
    scatter = ax.scatter(points[:,0], points[:,1], c=labels, 
                        cmap='tab10', s=5, alpha=0.6)
    ax.set_title(title, fontsize=10)
    ax.axis('on')
    return scatter

if __name__ == "__main__":
    # 初始化参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIFAR_10')
    parser.add_argument('--data', type=str, default=r'/home/users/pxx/workplace/Datasets/cifar10')
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--resize', type=int, default=32, help='location of the data')
    parser.add_argument('--load', type=str, default='FT_ACL', choices=['SSL', 'ACL', 'FT_ACL', 'Meta_ACL'])
    args = parser.parse_args()

    TXT_SAVE_PATH = os.path.join("result_tsne", f"CIFAR_10_variance_results_{args.load}.txt")
    PLOT_SAVE_PATH = os.path.join("result_tsne", f"comparative_tsne_CIFAR_10_{args.load}.png")

    # 获取数据加载器
    train_loader, val_loader, test_loader, num_classes, args = get_loader(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建可视化画布
    fig, axs = plt.subplots(2, 4, figsize=(10, 10))
    # fig.suptitle("Comparative t-SNE Visualization (CIFAR-10)", fontsize=16)

    # 结果收集
    results = []

    if args.load  == "FT_ACL":
        MODEL_CONFIGS = MODEL_CONFIGS_FT_ACL
    else:
        MODEL_CONFIGS = MODEL_CONFIGS_Meta_ACL

    # 遍历所有模型配置
    for idx, config in enumerate(MODEL_CONFIGS):
        # 模型处理
        model = load_model(config, args, num_classes, device)
        embeddings, labels = extract_embeddings(model, test_loader, device)
        
        # 方差计算
        intra, inter = compute_variances(embeddings, labels)
        results.append(f"{config['name']:<15} Intra: {intra:.2f} Inter: {inter:.2f}")
        
        # 可视化
        ax = axs[idx%2, idx//2]
        visualize_tsne(embeddings, labels, ax, config['name'])
    
    # 保存结果
    with open(TXT_SAVE_PATH, 'w') as f:
        f.write("\n".join(results))
    plt.savefig(PLOT_SAVE_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    time.sleep(10)