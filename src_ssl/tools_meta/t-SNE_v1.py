import os
import sys
sys.path.append(r'/home/users/pxx/workplace/5Adversarial/4Meta_AT_ResNet')
import torch
import argparse
from tqdm import tqdm
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from configs.experiment_config import load_SSL_model
from src_ssl.cifar10_dataset import CifarDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import time  # Add time module to introduce a pause

def extract_embeddings(model, loader):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for data in tqdm(loader):
            images = data[0].cuda()
            targets = data[1]
            emb = model(images, return_features=True).cpu().numpy()
            embeddings.append(emb)
            labels.append(targets.numpy())
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    return embeddings, labels

def plot_tsne(embeddings, labels, title, ax):
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000)
    tsne_results = tsne.fit_transform(embeddings)
    scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='tab10', s=5)
    ax.set_title(title)
    return scatter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Set up arguments (add relevant arguments as needed)
    parser.add_argument('--data_dir', type=str, default=r'/home/users/pxx/workplace/Datasets/GTSRB', metavar='D')
    args = parser.parse_args()

    # Load CIFAR-10 validation set
    data_transforms_val = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    val_dataset = CifarDataset(args.data_dir + '/val', transform=data_transforms_val)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, num_workers=8, shuffle=False)

    # Models (SimCLR, RoCL, RoCL-IP)
    model_names = ['simclr', 'rocl', 'rocl-ip']
    titles = ['(a) SimCLR', '(b) RoCL', '(c) RoCL-IP']
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    for i, model_name in enumerate(model_names):
        args.victim = model_name
        model, _, _ = load_SSL_model(args, device)
        embeddings, labels = extract_embeddings(model, val_loader)
        scatter = plot_tsne(embeddings, labels, titles[i], axs[i])

    fig.colorbar(scatter, ax=axs, orientation="horizontal", fraction=0.1, pad=0.1)
    plt.savefig("tsne_visualization_combined.png")
    plt.show()

    # Pause the display for 10 seconds before closing
    time.sleep(10)
