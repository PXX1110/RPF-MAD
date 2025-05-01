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
from torch.utils.tensorboard import SummaryWriter
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
    embeddings = []
    labels = []
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
    print(f"Labels shape: {labels.shape}")

    # 创建 TensorBoard 目录
    log_dir = f"runs/tensorboard_projector_{args.load}_{args.victim}_{args.dataset}"
    os.makedirs(log_dir, exist_ok=True)

    # 创建 metadata 文件
    metadata_path = os.path.join(log_dir, "metadata.tsv")
    if not os.path.exists(metadata_path):  # 只在文件不存在时创建
        with open(metadata_path, "w") as f:
            f.write("Index\tLabel\n")
            for i, label in enumerate(labels):
                f.write(f"{i}\t{label}\n")
    # 确保 labels 为字符串列表

    metadata_list = [str(label) for label in labels]
    # 初始化 TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir)
    # 添加嵌入数据
    writer.add_embedding(torch.tensor(embeddings), metadata=metadata_list)
    writer.close()
    print(f"Embedding visualization saved. Run TensorBoard with:\n tensorboard --logdir={log_dir}")
