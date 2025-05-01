# -*- coding: utf-8 -*-
"""
# @file name  : resnet-inference.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2020-06-23
# @brief      : inference demo
"""
import sys,os
sys.path.append(r'/home/users/pxx/workplace/5Adversarial/4Meta_AT_ResNet')
import time
import torch
import argparse
import numpy as np
from datetime import datetime
from configs.experiment_config import load_SSL_model
from src_ssl.tools_meta.train_common_tools import log
from src_ssl.tools_meta.attack_common_tools import process_img

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0, 1, 2') #, 1, 2, 3'
    parser.add_argument('--network', type=str, default='Resnet18',
                            help='choose which embedding network to use. Resnet18, AlexNet, EfficientNet')
    parser.add_argument('--attack', type=int, default=[True,False],    # True, False
                            help='used attacks')
    parser.add_argument('--dataset', type=str, default='CIFAR_10',
                            help='choose dataset to attack (CIFAR_10, MNIST, TinyImageNet)')
    parser.add_argument('--freezy-weight',  type=bool, default=False,
                            help='freezy_weight')
    parser.add_argument('--pre_dataset', default='CIFAR_10', choices=['CIFAR_10', 'ImageNet'])
    parser.add_argument('--victim', default='ACL', choices=['simclr', 'byol', 'dino', 'mocov3', 'mocov2plus',
                                 'nnclr', 'ressl', 'swav', 'vibcreg', 'wmse', 'deepclusterv2', 'robust_ACL','AInfoNCE', 'ACL'])
    parser.add_argument('--init', type=str, default='None', choices=['None', 'Roli', 'Meta'])
    parser.add_argument('--mode', type=str, default='SLF', choices=['SLF', 'ALF', 'AFF'])
    opt = parser.parse_args()
    
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    log_dir = os.path.join(BASE_DIR, "..", "..", "results_ssl", time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file_path = os.path.join(log_dir, "SSL_Clean_backbone_{}_{}_test_log.txt".format(opt.dataset, opt.attack[0]))    #YOPO_5_3  Clean  Attack
    # for i in range(0,5):
        # log_file_path = os.path.join(log_dir, "Meta_AT_Clean_{}_{}_test_log_attack_{}.txt".format(opt.dataset, opt.attack[0], i))
        # log(log_file_path, str(vars(opt)))
 
        # 2/5 load model
    if True:
        # model, _, data_dir = get_model(opt, device, i)
        model, _, _ = load_SSL_model(opt, device)
        data_dir = r"/home/users/pxx/workplace/Datasets/cifar10/test"
        model.eval()
        if True:
            acc = 0
            n = 0
            #  0~9
            for label_files in os.listdir(data_dir):
                label_file = os.path.join(data_dir, label_files)
                for path in os.listdir(label_file):
                    n += 1
                    # 1/5 load img
                    path_img = os.path.join(label_file, path)
                    # path_img = label_file
                    img_tensor, _ = process_img(path_img, opt)
                    img_tensor = img_tensor.cuda()
                    label_num = np.array([int(label_files)]).astype(np.int64)
                    label = torch.tensor(label_num).cuda()
                    # 3/5 inference  tensor --> vector
                    with torch.no_grad():
                        time_tic = time.time()
                        outputs = model(img_tensor)
                        time_toc = time.time()
                    # 4/5 index to class names
                    pred = torch.argmax(outputs, dim=1)
                    label = label.to(device)
                    acc += (pred ==label)
            acc_avg = acc/n
            log(log_file_path, 'inference acc: {}\n'.format(acc_avg))