import os
import sys
import wandb
sys.path.append(r'/home/users/pxx/workplace/5Adversarial/4Meta_AT_ResNet')
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tqdm import tqdm
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import argparse
from configs.experiment_config import get_loader, get_model_ACL, load_SSL_model, load_victim
sys.path.append(r'/home/users/pxx/workplace/5Adversarial/4Meta_AT_ResNet')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-epoch', type=int, default=50)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default='7')
    parser.add_argument('--dataset', type=str, default='gtsrb')
    parser.add_argument('--data', type=str, default=r'/home/users/pxx/workplace/Datasets/GTSRB', metavar='D')
    parser.add_argument('--resize', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--model', type=str, default=r'/home/users/pxx/workplace/5Adversarial/9_2Robust_SSL/RobustSSL_Benchmark-main/checkpoints/SSL_cifar10_r18_gtsrb_Meta_AdvCL_all_c/AFF/AFF_model_bestAT.pt')
    parser.add_argument('--pre_dataset', default='CIFAR_10', choices=['CIFAR_10', 'ImageNet'])
    parser.add_argument('--victim', default='AdvCL', choices=['simclr', 'byol', 'dino', 'mocov3', 'mocov2plus',
                                 'nnclr', 'ressl', 'swav', 'vibcreg', 'wmse', 'deepclusterv2', 'robust_ACL',
                                 'ACL', 'AdvCL', 'A-InfoNCE', 'DeACL','DynACL','DynACL++','DynACL-AIR','ACL_AIR','DynACL-AIR++','DynACL-RCS'
                            ,'Meta_ACL','Meta_AdvCL','Meta_A-InfoNCE','Meta_DeACL','Meta_DynACL','Meta_DynACL++','Meta_DynACL-AIR','Meta_DynACL-AIR++','Meta_DynACL-RCS']) 
    parser.add_argument('--init', type=str, default='None', choices=['None', 'Roli', 'Meta'])
    parser.add_argument('--mode', type=str, default='AFF', choices=['SLF', 'ALF', 'AFF'])
    parser.add_argument('--load', type=str, default='Meta_ACL', choices=['SSL', 'ACL', 'FT_ACL', 'Meta_ACL'])
    args = parser.parse_args()

    # 启动 WandB
    wandb.init(project="manifold_surface", name=f"{args.load}_{args.victim}_{args.dataset}")

    # 加载数据
    train_loader, val_loader, test_loader, num_classes, args = get_loader(args)

    # 加载模型
    if args.load == 'SSL':
        model = load_victim(args, device)
    elif args.load == 'ACL':
        model = get_model_ACL(args, device)
    elif args.load in ['FT_ACL', 'Meta_ACL']:
        from model.resnet_ssl import resnet18
        do_normalize = 0 if args.victim in ['AdvCL', 'A-InfoNCE','Meta_AdvCL', 'Meta_A-InfoNCE'] else 1
        model = resnet18(num_classes=num_classes, do_normalize=do_normalize)
        ckpt = torch.load(args.model)
        model.load_state_dict(ckpt['state_dict'])
        model.cuda()
    model.eval()

    # 提取嵌入
    embeddings, labels = [], []
    with torch.no_grad():
        for data in tqdm(test_loader):
            images = data[0].cuda()
            targets = data[1]
            emb = model(images, return_features=True).cpu().numpy()
            embeddings.append(emb)
            labels.append(targets.numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    print(f"Embeddings shape: {embeddings.shape}")

    # ===================== 数据预处理：PCA + 标准化 =====================
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # PCA 预处理，降至 50 维减少噪声，提高流形展开质量
    pca = PCA(n_components=50)
    embeddings_pca = pca.fit_transform(embeddings_scaled)

    # ============================ Isomap 3D 降维 ============================
    isomap = Isomap(n_components=3, n_neighbors=30)
    embeddings_isomap_3d = isomap.fit_transform(embeddings_pca)

    df_isomap_3d = pd.DataFrame(embeddings_isomap_3d, columns=["x", "y", "z"])
    df_isomap_3d["label"] = labels

    # ============================ 绘制 3D 曲面 ============================
    fig = go.Figure()

    # 添加曲面流形
    fig.add_trace(go.Mesh3d(
        x=df_isomap_3d["x"],
        y=df_isomap_3d["y"],
        z=df_isomap_3d["z"],
        colorbar_title='Manifold',
        colorscale="Viridis",
        opacity=0.5,
        alphahull=5  # 控制曲面包裹程度
    ))

    # 添加原始数据点
    fig.add_trace(go.Scatter3d(
        x=df_isomap_3d["x"],
        y=df_isomap_3d["y"],
        z=df_isomap_3d["z"],
        mode='markers',
        marker=dict(size=5, color=df_isomap_3d["label"], colorscale="Rainbow", opacity=0.8),
        name='Data Points'
    ))

    fig.update_layout(title="3D Isomap Manifold", margin=dict(l=0, r=0, b=0, t=40))

    # 上传到 WandB
    wandb.log({"3D Manifold Surface": wandb.Html(fig.to_html())})

    # 关闭 WandB
    wandb.finish()
    print("3D 流形嵌入图已生成，查看 WandB 结果！")
