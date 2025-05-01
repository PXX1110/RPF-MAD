# -*- coding: utf-8 -*-
"""
# @file name  : Meta_AT_train.py
# @author     : Xiaoxu Peng https://github.com/PXX1110
# @date       : 2023-09-01
"""
import sys,os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
sys.path.append(r'/home/users/pxx/workplace/5Adversarial/4Meta_AT_ResNet')
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
# from tensorboardX import SummaryWriter
from configs.experiment_config_clean import CRITERTION, get_model, get_dataset_train, get_dataset_train_without_class, load_SSL_model
from tools_meta.train_common_tools import LabelSmoothingLoss, ModelTrainer, set_gpu, log, Timer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set Optimizer
def set_optimizer(args, lr_max, params, optimizer, lr_schedule):   
    if  lr_schedule == 'cyclic':
        if optimizer == 'momentum':
            opt = torch.optim.SGD(params, lr=lr_max, momentum=0.9, weight_decay=args.weight_decay)
        elif optimizer == 'Adam':
            opt = torch.optim.Adam(params, lr=lr_max, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    else:
        if optimizer == 'momentum':
            opt = torch.optim.SGD(params, lr=lr_max, momentum=0.9, weight_decay=args.weight_decay)
        elif optimizer == 'Nesterov':
            opt = torch.optim.SGD(params, lr=lr_max, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
        elif optimizer == 'Adam':
            opt = torch.optim.Adam(params, lr=lr_max, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
        elif optimizer == 'AdamW':
            opt = torch.optim.AdamW(params, lr=lr_max, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    return opt

# Set Lr_schedule
def set_lr_schedule(args, lr_max):
    if args.lr_schedule == 'superconverge':
        lr_schedule = lambda t: np.interp([t], [0, args.num_epoch * 2 // 5, args.num_epoch], [0, lr_max, 0])[0]
    elif args.lr_schedule == 'piecewise':
        def lr_schedule(t, warm_up_lr = args.warmup_lr):
            if t <= 5:         #  60  100 
                if  warm_up_lr and t < args.warmup_lr_epoch:
                    return (t + 1.) / args.warmup_lr_epoch * lr_max
                else:
                    return lr_max
            if args.lrdecay == 'lineardecay':
                if t < 15:      #  60  105
                    return lr_max * 0.02 * (105 - t)
                else:
                    return 0.
            elif args.lrdecay == 'intenselr':
                if t < 102:
                    return lr_max / 10.
                else:
                    return lr_max / 100.
            elif args.lrdecay == 'looselr':
                if t < 150:
                    return lr_max / 10.
                else:
                    return lr_max / 100.
            elif args.lrdecay == 'base':
                if t <= 7:   # 90 105  
                    return lr_max / 10.
                else:
                    return lr_max / 100.
    elif args.lr_schedule == 'linear':
        lr_schedule = lambda t: np.interp([t], [0, args.num_epoch // 3, args.num_epoch * 2 // 3, args.num_epoch], [lr_max, lr_max, lr_max / 10, lr_max / 100])[0]
    elif args.lr_schedule == 'onedrop':
        def lr_schedule(t):
            if t < args.lr_drop_epoch:
                return lr_max
            else:
                return args.lr_one_drop
    elif args.lr_schedule == 'multipledecay':
        def lr_schedule(t):
            return lr_max - (t//(args.num_epoch//10))*(lr_max/10)
    elif args.lr_schedule == 'cosine': 
        def lr_schedule(t): 
            return lr_max * 0.5 * (1 + np.cos(t / args.num_epoch * np.pi))
    elif args.lr_schedule == 'cyclic':
        def lr_schedule(t, stepsize=18, min_lr=1e-5, max_lr=lr_max):
            # Scaler: we can adapt this if we do not want the triangular CLR
            scaler = lambda x: 1.
            # Additional function to see where on the cycle we are
            cycle = math.floor(1 + t / (2 * stepsize))
            x = abs(t / stepsize - 2 * cycle + 1)
            relative = max(0, (1 - x)) * scaler(cycle)
            return min_lr + (max_lr - min_lr) * relative
        
    return lr_schedule

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Group 1
    parser.add_argument('--num-epoch', type=int, default=10,  help='number of training epochs')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--save-epoch', type=int, default=5, help='frequency of model saving')
    parser.add_argument('--tensorboard_logdir', type=str, default='tensorboard', help='tensorboard model saving')
    parser.add_argument('--train-shot-way', type=int, default=2, help='number of attacks in one training episode')
    parser.add_argument('--train-query-way', type=int, default=1, help='number of attacks in one training episode')
    parser.add_argument('--val-shot-way', type=int, default=2, help='number of classes in one validation episode')
    parser.add_argument('--val-query-way', type=int, default=1, help='number of attacks in one validation episode')
    parser.add_argument('--train-shot', type=int, default=15, help='number of support examples per training class')
    parser.add_argument('--val-shot', type=int, default=5, help='number of support examples per validation class')
    parser.add_argument('--train-query', type=int, default=6, help='number of query examples per training class')
    parser.add_argument('--val-query', type=int, default=15, help='number of query examples per validation class')
    parser.add_argument('--train-episode', type=int, default=100, help='number of episodes per train')
    parser.add_argument('--val-episode', type=int, default=100, help='number of episodes per validation')
    parser.add_argument('--lamda', type=float, default=0.5, help='number of episodes per validation')
    parser.add_argument('--gpu', default='4,5,6') #, 1, 2, 3'
    parser.add_argument('--network', type=str, default='Resnet18', help='choose which embedding network to use. Resnet18, AlexNet, EfficientNet')
    parser.add_argument('--dataset', type=str, default='CIFAR_10', help='choose which meta_dataset to use. CIFAR_10, MNIST, TinyImageNet')             
    parser.add_argument('--attack', type=int, default=[True,False], help='used attacks')   # True, False             
    parser.add_argument('--attack_targeted', action='store_true', help='used targeted attacks')
    # Group 2
    parser.add_argument('--BNeval', action='store_true') # whether use eval mode for BN when crafting adversarial examples
    parser.add_argument('--earlystop-train', action='store_true') # whether use early stop, action='store_true'
    parser.add_argument('--earlystop-val', action='store_true') # whether use early stop, action='store_true'
    parser.add_argument('--labelsmooth', action='store_true') # whether use label smoothing , default=True , action='store_true'
    parser.add_argument('--labelsmoothvalue', default=0.1, type=float)
    parser.add_argument('--mixup', action='store_true')# whether use mixup , action='store_true'
    parser.add_argument('--mixup-alpha', default=1.0, type=float)
    parser.add_argument('--cutout', action='store_true')
    parser.add_argument('--cutout-len', default=14, type=int)
    parser.add_argument('--warmup_lr', action='store_true') # whether warm_up lr from 0 to max_lr in the first n epochs
    parser.add_argument('--warmup_lr_epoch', default=1, type=int)
    parser.add_argument('--lr-schedule', default='piecewise', choices=['superconverge', 'piecewise', 'linear', 'piecewisesmoothed', 'piecewisezoom', 'onedrop', 'multipledecay', 'cosine', 'cyclic'])
    parser.add_argument('--lr-max', default=0.01, type=float)
    parser.add_argument('--lr-one-drop', default=0.005, type=float)
    parser.add_argument('--lr-drop-epoch', default=100, type=int)
    parser.add_argument('--lrdecay', default='base', type=str, choices=['intenselr', 'base', 'looselr', 'lineardecay'])
    parser.add_argument('--optimizer', default='momentum')    #, choices=['momentum', 'Nesterov', 'SGD_GC', 'SGD_GCC', 'Adam', 'AdamW']
    parser.add_argument('--weight_decay', default=5e-4, type=float) # weight decay
    # Group 3
    parser.add_argument('--pre_dataset', default='CIFAR_10', choices=['CIFAR_10', 'ImageNet'])
    parser.add_argument('--victim', default='ACL', choices=['simclr', 'byol', 'dino', 'mocov3', 'mocov2plus',
                                 'nnclr', 'ressl', 'swav', 'vibcreg', 'wmse', 'deepclusterv2', 'robust_ACL', 'ACL','AInfoNCE','AInfoNCE_DRC'])
    parser.add_argument('--init', type=str, default='None', choices=['None', 'Roli', 'Meta'])
    parser.add_argument('--mode', type=str, default='SLF', choices=['SLF', 'ALF', 'AFF'])
    opt = parser.parse_args()

    # Set seed
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    set_gpu(opt.gpu)

    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    log_dir = os.path.join(BASE_DIR, "..", "results_ssl", time_str)
    # create save dir
    os.makedirs(log_dir, exist_ok=True) 
    log_file_path = os.path.join(log_dir, "{}_Meta_Meta_AT_train_with_clean_SAR.txt".format(opt.dataset))
    log(log_file_path, str(vars(opt)))
    network, _, _ = load_SSL_model(opt, device)
    network = nn.DataParallel(network).cuda()
    param_train = network.parameters()
    param_val = network.parameters()
    if opt.dataset == "TinyImageNet":
        (dloader_train, dloader_val) = get_dataset_train_without_class(opt)
    else:
        (dloader_train, dloader_val) = get_dataset_train(opt)

    opt_s = set_optimizer(opt, opt.lr_max, param_train, optimizer=opt.optimizer, lr_schedule=opt.lr_schedule)
    opt_q = set_optimizer(opt, opt.lr_max, param_val, optimizer=opt.optimizer, lr_schedule=opt.lr_schedule)
    lr_schedule_s = set_lr_schedule(opt, opt.lr_max)
    lr_schedule_q = set_lr_schedule(opt, opt.lr_max)
    # lambda_epoch = lambda e: 1.0 if e < 20 else (0.06 if e < 40 else 0.012 if e < 50 else (0.0024))
    # train_scheduler = torch.optim.lr_scheduler.LambdaLR(train_optimizer, lr_lambda=lambda_epoch, last_epoch=-1)
    # val_scheduler = torch.optim.lr_scheduler.LambdaLR(val_optimizer, lr_lambda=lambda_epoch, last_epoch=-1)
   
    if opt.labelsmooth:
        criterion = LabelSmoothingLoss(smoothing=opt.labelsmoothvalue)
    else:
        criterion = CRITERTION

    loss_rec = {"train": [], "valid": []}
    acc_rec = {"train": [], "valid": []}
    best_acc, best_epoch = 0.0, 0
    timer = Timer()
    # =========================== 断点恢复 ============================
    # path_checkpoint = r""
    # checkpoint = torch.load(path_checkpoint)
    # network.load_state_dict(checkpoint['model_state_dict'])
    # val_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # start_epoch = checkpoint['epoch']
    # epoch = val_scheduler.last_epoch
    # loss_rec = checkpoint['loss_rec']
    # acc_rec = checkpoint['acc_rec']

    print("Start Training, {}!".format(opt.network))  # 定义遍历数据集的次数
    for epoch in range(1, opt.num_epoch + 1):
        # Train on the training split
        # Fetch the current epoch's learning rate
        train_loss_avg, train_acc_avg, lr_s, optimizer, network = ModelTrainer.train_ssl_clean_SAR(epoch ,dloader_train, network, criterion, opt_s, lr_schedule_s, opt.num_epoch, log_file_path, log_dir, opt)
        log(log_file_path, 'Train Epoch: {}\tLearning Rate: {:.4f}'.format(epoch, lr_s))
        loss_rec["train"].append(train_loss_avg)
        acc_rec["train"].append(train_acc_avg)

        if (epoch) % opt.save_epoch == 0 or epoch >= 5:
            print("Waiting Test!")
            val_loss_avg, val_acc_avg = ModelTrainer.valid_ssl_multi(epoch, dloader_val, network, criterion, optimizer, opt.num_epoch, log_file_path, log_dir, opt)
            log(log_file_path, 'Test Epoch: {}\tLearning Rate: {:.4f}'.format(epoch, lr_s))
            # val_scheduler.step()
            loss_rec["valid"].append(val_loss_avg)
            acc_rec["valid"].append(val_acc_avg)
            if val_acc_avg > best_acc:
                best_acc = val_acc_avg
                best_epoch = epoch
                checkpoint = {"state_dict": network.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss_rec": loss_rec,
                            "acc_rec": acc_rec,
                            "epoch": epoch}
                path_checkpoint = os.path.join(log_dir,"./best_checkpint.pkl".format(epoch))
                torch.save(checkpoint, path_checkpoint)

            checkpoint = {"state_dict": network.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss_rec": loss_rec,
                        "acc_rec": acc_rec,
                        "epoch": epoch}
            path_checkpoint = os.path.join(log_dir,"./checkpint_{}_epoch.pkl".format(epoch))
            torch.save(checkpoint, path_checkpoint)

    # 记录最佳测试分类准确率并写入best_acc.txt文件中
    log(log_file_path, " done ~~~~ {}, best acc: {} in :{} epochs. ".format(datetime.strftime(datetime.now(), '%m-%d_%H-%M'), best_acc, best_epoch))
    log(log_file_path, 'Elapsed Time: {}/{}\n'.format(timer.measure(), timer.measure(epoch / float(opt.num_epoch))))

