# -*- coding: utf-8 -*-
"""
# @file name  : train_resnet.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2020-06-23
# @brief      : resnet training on cifar10
"""
import sys,os
import numpy as np
sys.path.append('/home/users/pxx/workplace/5Adversarial/4Meta_AT_ResNet')
import torch
import argparse
import torch.optim as optim
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.nn as nn
from pathlib import Path
from cifar10_dataset import CifarDataset
from configs.attack_list_config import get_attack_CIFAER10, get_attack_GTSRB, get_attack_MNIST, get_attack_TinyImageNet
from configs.experiment_config import CIFAR_CLASS_NAMES, GTSRB_CLASS_NAMES, MNIST_CLASS_NAMES, TINYIMAGENET_CLASS_NAMES, get_model, load_victim
from src_sl.tools_meta.attack_common_tools import get_classifier,  test_attack_samples
from src_sl.tools_meta.train_common_tools import set_gpu, log
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Group 1
    parser.add_argument('--num-epoch', type=int, default=50,
                            help='number of training epochs')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default='7') #, 1, 2, 3'
    parser.add_argument('--network', type=str, default='Resnet18',
                            help='choose which embedding network to use. Resnet18, AlexNet, EfficientNet')
    parser.add_argument('--attack_targeted', action='store_true',
                            help='used targeted attacks')
    parser.add_argument('--attack', type=int, default=[True,False],    # True, False
                            help='used attacks')
    parser.add_argument('--dataset', type=str, default='gtsrb',
                            help='choose which classification head to use. CIFAR_10, MNIST, TinyImageNet')
    parser.add_argument('--data', type=str, default=r'/home/users/pxx/workplace/Datasets/GTSRB', metavar='D')  #/home/users/pxx/workplace/Datasets/GTSRB
    parser.add_argument('--resize', type=int, default=32, help='location of the data')
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--model', type=str, default=r'/home/users/pxx/workplace/5Adversarial/4Meta_AT_ResNet/src_ssl/victims/CIFAR_10/simclr/simclr-cifar10-b30xch14-ep=999.ckpt')
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
    parser.add_argument('--mode', type=str, default='SLF', choices=['SLF', 'ALF', 'AFF'])
    parser.add_argument('--load', type=str, default='SSL', choices=['SSL', 'ACL', 'FT_ACL', 'Meta_ACL'])
    opt = parser.parse_args()

    log_dir = os.path.join(BASE_DIR, "..", "results_gtsrb", "Attacked_Resnet18_{}_{}").format(opt.dataset, opt.victim)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file_path = os.path.join(log_dir, "test_train_log.txt")
    log(log_file_path, str(vars(opt)))

    if opt.load == 'SSL':
        # load SSL model
        encoder = load_victim(opt, device)
        classifier_path = "/home/users/pxx/workplace/5Adversarial/9_2Robust_SSL/AdvEncoder-master/victims/cifar10/simclr/clean_model/gtsrb/simclr_cifar10_gtsrb_79.0985_16.pth"
        classifier = torch.load(classifier_path)
        model = nn.Sequential(encoder, classifier)
        model.to(device)
        data_dir = r"/home/users/pxx/workplace/Datasets/GTSRB/val" #/home/users/pxx/workplace/Datasets/MADS/MAD-G
    elif opt.load == 'AT':
        model, checkpoint, data_dir = get_model(opt, device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)  # 选择优化器
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()

    if opt.dataset == 'CIFAR_10':
        classifier = get_classifier(model=model, optimizer=optimizer, input_shape=(3, 32, 32), nb_classes=len(CIFAR_CLASS_NAMES))
        cls_n = CIFAR_CLASS_NAMES
        ATKS = get_attack_CIFAER10(classifier, model)
    elif opt.dataset == 'TinyImageNet':
        classifier = get_classifier(model=model, optimizer=optimizer, input_shape=(3, 224, 224), nb_classes=len(TINYIMAGENET_CLASS_NAMES))
        cls_n = TINYIMAGENET_CLASS_NAMES
        ATKS = get_attack_TinyImageNet(classifier, model)
    elif opt.dataset == 'MNIST':
        # classifier = get_classifier(model=model, optimizer=optimizer, input_shape=(3, 28, 28), nb_classes=len(MNIST_CLASS_NAMES))
        classifier = get_classifier(model=model, optimizer=optimizer, input_shape=(3, 32, 32), nb_classes=len(MNIST_CLASS_NAMES))
        cls_n = MNIST_CLASS_NAMES
        ATKS = get_attack_MNIST(classifier, model)
    elif opt.dataset == 'gtsrb':
        classifier = get_classifier(model=model, optimizer=optimizer, input_shape=(3, 32, 32), nb_classes=len(GTSRB_CLASS_NAMES))
        cls_n = GTSRB_CLASS_NAMES
        ATKS = get_attack_GTSRB(classifier, model)
    print("Start Attacking {}!".format(opt.network))  # 定义遍历数据集的次数

    # transform_list = [lambda x: np.asarray(x), transforms.ToTensor()]
    # transform_chain = transforms.Compose(transform_list)
    # item = CifarDataset(data_dir=r"/home/users/pxx/workplace/5Adversarial/Meta_Datasets/new/new_CIFAR_10/val" , transform=transform_chain)   #False    True
    # test_loader = data.DataLoader(item, batch_size=128, shuffle=False, num_workers=0)

    ATKSID = {key: [ATKS[key]] for key in ATKS}
    for i in range(11, 12):

        atk_key = list(ATKSID)[i]
        atk = ATKS[atk_key]
        # acc, adv_acc = test_attack_batch_samples(i, atk_key, atk, cls_n, model, test_loader, log_dir)
        acc, adv_acc = test_attack_samples(i, atk_key, atk, cls_n, model, data_dir, log_dir, opt)
        log(log_file_path, str("Attack {} Done!".format(i)) + str("acc:{}".format(acc)) + str("adv_acc:{}".format(adv_acc.cpu().numpy())))

    print("Attack {} Done!".format(opt.dataset))
