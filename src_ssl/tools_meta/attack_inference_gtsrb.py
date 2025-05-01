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
from configs.experiment_config import get_model, get_model_ACL, load_SSL_model, load_victim
from train_common_tools import log
from attack_common_tools import process_img

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='4') #, 1, 2, 3'
    parser.add_argument('--network', type=str, default='Resnet18',
                            help='choose which embedding network to use. Resnet18, AlexNet, EfficientNet')
    parser.add_argument('--attack', type=int, default=[True,False],    # True, False
                            help='used attacks')
    parser.add_argument('--model', type=str, default=r'/home/users/pxx/workplace/5Adversarial/9_2Robust_SSL/RobustSSL_Benchmark-main/checkpoints/SSL_cifar10_r18_gtsrb_Meta_DynACL++/AFF/AFF_model_bestAT.pt')
    parser.add_argument('--dataset', type=str, default='gtsrb',
                            help='choose dataset to attack (CIFAR_10, MNIST, TinyImageNet, gtsrb)')
    parser.add_argument('--freezy-weight',  type=bool, default=False,
                            help='freezy_weight')
    parser.add_argument('--pre_dataset', default='CIFAR_10', choices=['CIFAR_10', 'ImageNet'])
    parser.add_argument('--victim', default='Meta_DynACL++', choices=['simclr', 'byol', 'dino', 'mocov3', 'mocov2plus',
                                 'nnclr', 'ressl', 'swav', 'vibcreg', 'wmse', 'deepclusterv2', 'robust_ACL',
                                 'ACL', 'AdvCL', 'A-InfoNCE', 'DeACL','DynACL','DynACL++','DynACL-AIR','ACL_AIR','DynACL-AIR++','DynACL-RCS'
                            ,'Meta_ACL','Meta_AdvCL','Meta_A-InfoNCE','Meta_DeACL','Meta_DynACL','Meta_DynACL++','Meta_DynACL-AIR','Meta_DynACL-AIR++','Meta_DynACL-RCS']) 
    parser.add_argument('--init', type=str, default='None', choices=['None', 'Roli', 'Meta'])
    parser.add_argument('--mode', type=str, default='AFF', choices=['SLF', 'ALF', 'AFF'])
    parser.add_argument('--load', type=str, default='Meta_ACL', choices=['SSL', 'ACL', 'FT_ACL', 'Meta_ACL'])
    args = parser.parse_args()
    
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    log_dir = os.path.join(BASE_DIR, "..", "..", "results_ssl", time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file_path = os.path.join(log_dir, "{}_{}_{}_{}_{}_test_log.txt".format(args.victim, args.mode, args.load, args.dataset, args.attack[0]))    #YOPO_5_3  Clean  Attack
    
    if args.load == 'SSL':
        # load SSL model
        model = load_victim(args, device)
    elif args.load == 'ACL':
        # load ACL model
        model = get_model_ACL(args, device)
    elif args.load == 'FT_ACL' or args.load == 'Meta_ACL':
        # load Fine_tuned ACL model or Meta_ACL
        from model.resnet_ssl import resnet18
        if args.victim in ['AdvCL', 'A-InfoNCE','Meta_AdvCL', 'Meta_A-InfoNCE']:
            do_normalize = 0
        else: 
            do_normalize = 1
        model = resnet18(num_classes=43, do_normalize=do_normalize)
        ckpt = torch.load(args.model)
        model.load_state_dict(ckpt['state_dict'])
        model.cuda()
    model.eval()
    # for i in range(0,5):
        # log_file_path = os.path.join(log_dir, "Meta_AT_Clean_{}_{}_test_log_attack_{}.txt".format(opt.dataset, opt.attack[0], i))
        # log(log_file_path, str(vars(opt)))
 
        # 2/5 load model
    if True:
        data_dir = r"results_gtsrb/MAD-G"
        # data_dir = r"/home/users/pxx/workplace/Datasets/MADS/MAD-C"
        model.eval()

        for attack_files in os.listdir(data_dir):                     #  0~49   train
            # attack_file = os.path.join(data_dir, attack_files)
            attack_file = os.path.join(data_dir, attack_files,"test")
        # if True:
            acc = 0
            n = 0
            for label_files in os.listdir(attack_file):               #  0~9
                label_file = os.path.join(attack_file, label_files)
                for path in os.listdir(label_file):
                    n += 1
                    # 1/5 load img
                    path_img = os.path.join(label_file, path)
                    # path_img = label_file
                    img_tensor, _ = process_img(path_img, args)
                    img_tensor = img_tensor.cuda()
                    label_num = np.array([int(label_files)]).astype(np.int64)
                    # label_num = np.array([int(attack_files)]).astype(np.int64)
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
            log(log_file_path, 'attack {},inference acc: {}\n'.format(attack_files,acc_avg))
