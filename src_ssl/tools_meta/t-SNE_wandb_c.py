import os
import sys
import wandb
sys.path.append(r'/home/users/pxx/workplace/5Adversarial/4Meta_AT_ResNet')

import torch
import argparse
import numpy as np
import pandas as pd
import plotly.express as px
from tqdm import tqdm
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, SpectralEmbedding
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap

from configs.experiment_config import get_loader, get_model_ACL, load_SSL_model, load_victim
from src_ssl.cifar10_dataset import CifarDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-epoch', type=int, default=50)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default='7')
    parser.add_argument('--dataset', type=str, default='CIFAR_10')
    parser.add_argument('--data', type=str, default=r'/home/users/pxx/workplace/Datasets/cifar10', metavar='D')
    parser.add_argument('--resize', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--model', type=str, default=r'results_ssl/CIFAR_10_Meta_Multi_AT_train_0.001_0.0001_SGD_AdvCL_80_37/best_checkpint.pkl')
    parser.add_argument('--pre_dataset', default='CIFAR_10', choices=['CIFAR_10', 'ImageNet'])
    parser.add_argument('--victim', default='Meta_AdvCL', choices=['simclr', 'byol', 'dino', 'mocov3', 'mocov2plus',
                                 'nnclr', 'ressl', 'swav', 'vibcreg', 'wmse', 'deepclusterv2', 'robust_ACL',
                                 'ACL', 'AdvCL', 'A-InfoNCE', 'DeACL','DynACL','DynACL++','DynACL-AIR','ACL_AIR','DynACL-AIR++','DynACL-RCS'
                            ,'Meta_ACL','Meta_AdvCL','Meta_A-InfoNCE','Meta_DeACL','Meta_DynACL','Meta_DynACL++','Meta_DynACL-AIR','Meta_DynACL-AIR++','Meta_DynACL-RCS']) 
    parser.add_argument('--init', type=str, default='None', choices=['None', 'Roli', 'Meta'])
    parser.add_argument('--mode', type=str, default='AFF', choices=['SLF', 'ALF', 'AFF'])
    parser.add_argument('--load', type=str, default='Meta_ACL', choices=['SSL', 'ACL', 'FT_ACL', 'Meta_ACL'])
    args = parser.parse_args()

    # 启动 WandB 运行
    wandb.init(project="manifold_smoothing", name=f"{args.load}_{args.victim}_{args.dataset}")

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

    # ============================ t-SNE 2D 可视化 ============================
    tsne_2d = TSNE(n_components=2, perplexity=50, learning_rate=100, n_iter=3000, early_exaggeration=5)
    embeddings_tsne_2d = tsne_2d.fit_transform(embeddings_pca)
    df_tsne_2d = pd.DataFrame(embeddings_tsne_2d, columns=["x", "y"])
    df_tsne_2d["label"] = labels

    fig_tsne_2d = px.scatter(df_tsne_2d, x="x", y="y", color=df_tsne_2d["label"].astype(str), title="t-SNE 2D")
    wandb.log({"t-SNE 2D": wandb.Html(fig_tsne_2d.to_html())})

    # ============================ t-SNE 3D 可视化 ============================
    tsne_3d = TSNE(n_components=3, perplexity=50, learning_rate=100, n_iter=3000, early_exaggeration=5)
    embeddings_tsne_3d = tsne_3d.fit_transform(embeddings_pca)
    df_tsne_3d = pd.DataFrame(embeddings_tsne_3d, columns=["x", "y", "z"])
    df_tsne_3d["label"] = labels

    fig_tsne_3d = px.scatter_3d(df_tsne_3d, x="x", y="y", z="z", color=df_tsne_3d["label"].astype(str), title="t-SNE 3D")
    wandb.log({"t-SNE 3D": wandb.Html(fig_tsne_3d.to_html())})

    # ============================ UMAP 2D 可视化 ============================
    umap_2d = umap.UMAP(n_components=2, n_neighbors=50, min_dist=0.3)
    embeddings_umap_2d = umap_2d.fit_transform(embeddings_pca)
    df_umap_2d = pd.DataFrame(embeddings_umap_2d, columns=["x", "y"])
    df_umap_2d["label"] = labels

    fig_umap_2d = px.scatter(df_umap_2d, x="x", y="y", color=df_umap_2d["label"].astype(str), title="UMAP 2D")
    wandb.log({"UMAP 2D": wandb.Html(fig_umap_2d.to_html())})

    # ============================ Isomap 2D 可视化 ============================
    isomap = Isomap(n_components=2, n_neighbors=50)
    embeddings_isomap = isomap.fit_transform(embeddings_pca)
    df_isomap = pd.DataFrame(embeddings_isomap, columns=["x", "y"])
    df_isomap["label"] = labels

    fig_isomap = px.scatter(df_isomap, x="x", y="y", color=df_isomap["label"].astype(str), title="Isomap 2D")
    wandb.log({"Isomap 2D": wandb.Html(fig_isomap.to_html())})

    # ============================ LLE 2D 可视化 ============================
    lle = LocallyLinearEmbedding(n_components=2, n_neighbors=50)
    embeddings_lle = lle.fit_transform(embeddings_pca)
    df_lle = pd.DataFrame(embeddings_lle, columns=["x", "y"])
    df_lle["label"] = labels

    fig_lle = px.scatter(df_lle, x="x", y="y", color=df_lle["label"].astype(str), title="LLE 2D")
    wandb.log({"LLE 2D": wandb.Html(fig_lle.to_html())})

    # ============================ Laplacian Eigenmaps 2D ============================
    laplacian = SpectralEmbedding(n_components=2, n_neighbors=50, affinity="nearest_neighbors")
    embeddings_laplacian = laplacian.fit_transform(embeddings_pca)
    df_laplacian = pd.DataFrame(embeddings_laplacian, columns=["x", "y"])
    df_laplacian["label"] = labels

    fig_laplacian = px.scatter(df_laplacian, x="x", y="y", color=df_laplacian["label"].astype(str), title="Laplacian Eigenmaps 2D")
    wandb.log({"Laplacian Eigenmaps 2D": wandb.Html(fig_laplacian.to_html())})

    # 关闭 WandB
    wandb.finish()
    print("t-SNE / UMAP / Isomap / LLE / Laplacian Eigenmaps 可视化完成，查看 WandB 结果！")
