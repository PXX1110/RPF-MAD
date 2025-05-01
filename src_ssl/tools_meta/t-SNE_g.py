import os
import sys
sys.path.append(r'/home/users/pxx/workplace/5Adversarial/4Meta_AT_ResNet')
import torch
import argparse
from tqdm import tqdm
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from configs.experiment_config import get_loader, get_model_ACL, load_SSL_model, load_victim
from src_ssl.cifar10_dataset import CifarDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import time  # Add time module to introduce a pause

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Group 1
    parser.add_argument('--num-epoch', type=int, default=50,
                            help='number of training epochs')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default='7') #, 1, 2, 3'
    parser.add_argument('--dataset', type=str, default='gtsrb',
                            help='choose which classification head to use. CIFAR_10, MNIST, TinyImageNet')
    parser.add_argument('--data', type=str, default=r'/home/users/pxx/workplace/Datasets/GTSRB', metavar='D')  #/home/users/pxx/workplace/Datasets/cifar10
    parser.add_argument('--resize', type=int, default=32, help='location of the data')
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--model', type=str, default=r'/home/users/pxx/workplace/5Adversarial/9_2Robust_SSL/RobustSSL_Benchmark-main/checkpoints/SSL_cifar10_r18_gtsrb_meta_0.01_cross_LAMBDA3_1/AFF/AFF_model_bestAT.pt')
    # Group 2
    parser.add_argument('--earlystop-train', default=False) # whether use early stop, action='store_true'
    parser.add_argument('--earlystop-val', default=True) # whether use early stop, action='store_true'
    parser.add_argument('--labelsmooth', default=True) # whether use label smoothing  , action='store_true'
    parser.add_argument('--labelsmoothvalue', default=0.2, type=float)
    parser.add_argument('--mixup', default=True, action='store_true')# whether use mixup , action='store_true'
    parser.add_argument('--mixup-alpha', default=1.4, type=float)
    parser.add_argument('--cutout', action='store_true')
    parser.add_argument('--cutout-len', default=14, type=int)
    
    # Group 3
    parser.add_argument('--pre_dataset', default='CIFAR_10', choices=['CIFAR_10', 'ImageNet'])
    parser.add_argument('--victim', default='ACL', choices=['simclr', 'byol', 'dino', 'mocov3', 'mocov2plus',
                                 'nnclr', 'ressl', 'swav', 'vibcreg', 'wmse', 'deepclusterv2', 'robust_ACL',
                                 'ACL', 'Meta_ACL', 'AdvCL', 'A-InfoNCE', 'DeACL', 'DynACL', 'DynACL++', 'DynACL-AIR', 'DynACL-AIR++', 'DynACL-RCS'])
    parser.add_argument('--init', type=str, default='None', choices=['None', 'Roli', 'Meta'])
    parser.add_argument('--mode', type=str, default='AFF', choices=['SLF', 'ALF', 'AFF'])
    parser.add_argument('--load', type=str, default='Meta_ACL', choices=['SSL', 'ACL', 'FT_ACL', 'Meta_ACL'])
    args = parser.parse_args()

    # Load CIFAR-10 validation set
    train_loader, val_loader, test_loader, num_classes, args = get_loader(args)

    # Assume `model` is the trained model (SimCLR, RoCL, or RoCL-IP) without the last classification layer
    if args.load == 'SSL':
        # load SSL model
        model = load_victim(args, device)
    elif args.load == 'ACL':
        # load ACL model
        model = get_model_ACL(args, device)
    elif args.load == 'FT_ACL' or args.load == 'Meta_ACL':
        # load Fine_tuned ACL model or Meta_ACL
        from model.resnet_ssl import resnet18
        if args.victim in ['AdvCL', 'A-InfoNCE']:
            do_normalize = 0
        else: 
            do_normalize = 1
        model = resnet18(num_classes=num_classes, do_normalize=do_normalize)
        ckpt = torch.load(args.model)
        model.load_state_dict(ckpt['state_dict'])
        model.cuda()
    model.eval()

    # Extract embeddings
    embeddings = []
    labels = []

    with torch.no_grad():
        for counter, data in enumerate(tqdm(val_loader)):
            images = data[0].cuda()
            targets = data[1]
            emb = model(images, return_features=True).cpu().numpy()
            embeddings.append(emb)
            labels.append(targets.numpy())

    # Concatenate all embeddings and labels
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000)
    tsne_results = tsne.fit_transform(embeddings)

    # Plot the t-SNE visualization
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='tab10', s=5)
    plt.colorbar(scatter)
    plt.title(f"t-SNE Visualization of {args.load}_{args.mode} Embeddings for {args.dataset}")
    plt.savefig(f"tsne_visualization_meta_ACL_cross_LAMBDA3_1_{args.pre_dataset}_{args.dataset}_{args.load}.png")
    plt.show()
    # Pause the display for 10 seconds before closing
    time.sleep(10)