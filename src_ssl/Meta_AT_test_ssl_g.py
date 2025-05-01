# -*- coding: utf-8 -*-
"""
# @file name  : Meta_AT_test.py
# @author     : Xiaoxu Peng https://github.com/PXX1110
# @date       : 2023-09-01
"""
import sys,os
sys.path.append(r'/home/users/pxx/workplace/5Adversarial/4Meta_AT_ResNet')
import torch
import torch.nn as nn
import argparse
import torch.optim as optim
from datetime import datetime
from typing import OrderedDict
from configs.experiment_config import CRITERTION, get_dataset_test_single, get_model_ACL, load_victim
from configs.attack_label_config import get_attack_train, get_attack_val, get_attack_test
from tools_meta.train_common_tools import ModelTrainer, set_gpu, log, Timer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-shot-way', type=int, default=1,
                            help='number of classes in one validation episode')
    parser.add_argument('--test-query-way', type=int, default=1,
                            help='number of attacks in one validation episode')
    parser.add_argument('--test-shot', type=int, default=1,
                            help='number of support examples per validation class')
    parser.add_argument('--test-query', type=int, default=15,
                            help='number of query examples per validation class')
    parser.add_argument('--test-episode', type=int, default=100,
                            help='number of episodes per validation')
    parser.add_argument('--episodes-per-batch', type=int, default=1,
                            help='number of episodes per batch')
    parser.add_argument('--gpu', default='5') # 0, 1, 2, 3, 4, 5, 6, 7 
    parser.add_argument('--network', type=str, default='Resnet18',
                            help='choose which embedding network to use. Resnet18, AlexNet, EfficientNet')
    parser.add_argument('--model', type=str, default=r'/home/users/pxx/workplace/5Adversarial/9_2Robust_SSL/RobustSSL_Benchmark-main/checkpoints/DynACL_AIR++_cifar10_r18_gtsrb_2/AFF/AFF_model_bestAT.pt')
    parser.add_argument('--dataset', type=str, default='gtsrb',
                            help='choose which classification head to use. CIFAR_10, MNIST, TinyImageNet, ')
    parser.add_argument('--attack', type=int, default=[True,False],    # True, False
                            help='used attacks')
    parser.add_argument('--phase', type=str, default="test",          # train, val, test
                            help='used attacks')
    parser.add_argument('--attack_targeted', action='store_true',
                            help='used targeted attacks')
    parser.add_argument('--pre_dataset', default='CIFAR_10', choices=['CIFAR_10', 'ImageNet'])
    parser.add_argument('--victim', default='DynACL-AIR++', choices=['simclr', 'byol', 'dino', 'mocov3', 'mocov2plus',
                                 'nnclr', 'ressl', 'swav', 'vibcreg', 'wmse', 'deepclusterv2', 'robust_ACL',
                                 'ACL', 'AdvCL', 'A-InfoNCE', 'DeACL','DynACL','DynACL++','DynACL-AIR','ACL_AIR','DynACL-AIR++','DynACL-RCS'
                            ,'Meta_ACL','Meta_AdvCL','Meta_A-InfoNCE','Meta_DeACL','Meta_DynACL','Meta_DynACL++','Meta_DynACL-AIR','Meta_DynACL-AIR++','Meta_DynACL-RCS']) 
    parser.add_argument('--init', type=str, default='None', choices=['None', 'Roli', 'Meta'])
    parser.add_argument('--mode', type=str, default='AFF', choices=['SLF', 'ALF', 'AFF'])
    parser.add_argument('--load', type=str, default='FT_ACL', choices=['SSL', 'ACL', 'FT_ACL', 'Meta_ACL'])
    opt = parser.parse_args()

    set_gpu(opt.gpu)
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    log_dir = os.path.join(BASE_DIR, "..", "results_ssl", "Gtsrb_wiogl_Strong_MCL_robust_best_new_{}_cc".format(opt.load), "{}_{}".format(opt.victim, opt.phase))
    # create save dir
    os.makedirs(log_dir, exist_ok=True) 
    log_file_path = os.path.join(log_dir, "GtsrbMCL_{}_{}_attack_test_log.txt".format(opt.dataset, opt.phase))
    log(log_file_path, str(vars(opt)))    

    if opt.phase == "train":   
        ATK = get_attack_train
    elif opt.phase == "val":
        ATK = get_attack_val
    else:
        ATK = get_attack_test

    # for i in range(6,7):
    for i in range(len(ATK(opt.attack, opt.dataset))):
        dloader_test = get_dataset_test_single(opt, phase =opt.phase, attack_num=i)
        if opt.load == 'SSL':
            # load SSL model
            model = load_victim(opt, device)
        elif opt.load == 'ACL':
            # load ACL model
            model = get_model_ACL(opt, device)
        elif opt.load == 'FT_ACL' or opt.load == 'Meta_ACL':
            # load Fine_tuned ACL model or Meta_ACL
            from model.resnet_ssl import resnet18
            if opt.victim in ['AdvCL', 'A-InfoNCE','Meta_AdvCL', 'Meta_A-InfoNCE']:
                do_normalize = 0
            else: 
                do_normalize = 1
            model = resnet18(num_classes=43, do_normalize=do_normalize)
            ckpt = torch.load(opt.model)
            model.load_state_dict(ckpt['state_dict'])
            model.cuda()
        model.eval()
        test_optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.09, weight_decay=1e-4, nesterov=True)  # 选择优化器
        # test_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        timer = Timer()

        print("Waiting Test!")
        test_loss, test_acc = ModelTrainer.test_ssl(dloader_test, model, CRITERTION, test_optimizer, log_file_path, log_dir, i, opt)
        log(log_file_path, 'Elapsed Time: {}\n'.format(timer.measure()))