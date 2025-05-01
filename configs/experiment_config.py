import copy
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
from typing import OrderedDict
from model.alexnet import AlexNet
from model.efficientnet_pytorch.model import EfficientNet
# from model.resnet_flc import resnet18
from model.resnet_s import ResNet18, ResNet50
from model.preactresnet import create_network
from pathlib import Path
from torchvision import datasets, transforms
# Global config for the experiment
# Basic experiments parameters
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CIFAR_CLASS_NAMES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
MNIST_CLASS_NAMES = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
TINYIMAGENET_CLASS_NAMES = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24",
                            "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", 
                            "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70",
                            "71", "72", "73", "74", "75", "76", "77", "78", "79", "80", "81", "82", "83", "84", "85", "86", "87", "88", "89", "90", "91", "92", "93",
                            "94", "95", "96", "97", "98", "99", "100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "110", "111", "112", "113", "114",
                            "115", "116", "117", "118", "119", "120", "121", "122", "123", "124", "125", "126", "127", "128", "129", "130", "131", "132", "133", "134",
                            "135", "136", "137", "138", "139", "140", "141", "142", "143", "144", "145", "146", "147", "148", "149", "150", "151", "152", "153", "154",
                            "155", "156", "157", "158", "159", "160", "161", "162", "163", "164", "165", "166", "167", "168", "169", "170", "171", "172", "173", "174",
                            "175", "176", "177", "178", "179", "180", "181", "182", "183", "184", "185", "186", "187", "188", "189", "190", "191", "192", "193", "194",
                            "195", "196", "197", "198", "199")
GTSRB_CLASS_NAMES = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24",
                    "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42")

CIFAR_DIR = r"E:\Workplace\Datasets\cifar-10\cifar-10-batches-py"
MNIST_DIR = r"E:\Workplace\Datasets\MNIST\raw"

CRITERTION = nn.CrossEntropyLoss()

def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)

class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)

# DeACL
def load_BN_checkpoint_DeACL(args,state_dict):
    new_state_dict = {}
    new_state_dict_normal = {}
    for k, v in state_dict.items():
        if 'backbone.' in k:
            k = k.replace('backbone.', '')
        new_state_dict_normal[k] = v
        new_state_dict[k] = v
    return new_state_dict

# AdvCL and A-InfoNCE
def load_BN_checkpoint_AdvCL(args, state_dict):  
    new_state_dict = {}
    new_state_dict_normal = {}
    for k, v in state_dict.items():
        if 'downsample.bn.bn_list.0' in k:
            k = k.replace('downsample.bn.bn_list.0', 'downsample.0')
            new_state_dict_normal[k] = v
        elif 'downsample.bn.bn_list.1' in k:
            k = k.replace('downsample.bn.bn_list.1', 'downsample.1')
            new_state_dict[k] = v
        elif '.bn_list.0' in k:
            k = k.replace('.bn_list.0', '')
            new_state_dict_normal[k] = v
        elif '.bn_list.1' in k:
            k = k.replace('.bn_list.1', '')
            new_state_dict[k] = v
        elif 'downsample.conv' in k:
            k = k.replace('downsample.conv', 'downsample.0')
            new_state_dict_normal[k] = v
            new_state_dict[k] = v
        else:
            new_state_dict_normal[k] = v
            new_state_dict[k] = v
    
    return new_state_dict

# DeACL
def load_BN_checkpoint_lora(state_dict):
    new_state_dict = {}
    new_state_dict_normal = {}
    for k, v in state_dict.items():
        if 'conv1.weight' in k and k != 'conv1.weight':
            k = k.replace('conv1.weight', 'conv1.conv.weight')
        elif 'conv2.weight' in k:
            k = k.replace('conv2.weight', 'conv2.conv.weight')
        elif 'conv3.weight' in k:
            k = k.replace('conv3.weight', 'conv3.conv.weight')
        elif 'conv4.weight' in k:
            k = k.replace('conv4.weight', 'conv4.conv.weight')
        elif 'downsample.conv.weight' in k:
            k = k.replace('downsample.conv.weight', 'downsample.conv.conv.weight')
        new_state_dict_normal[k] = v
        new_state_dict[k] = v
    return new_state_dict, new_state_dict_normal

def cvt_state_dict(state_dict, args, num_classes):
    state_dict_new = copy.deepcopy(state_dict)
    if args.init == "Roli":
        if args.bnNameCnt >= 0:
            new_state_dict = OrderedDict()
            for name, item in state_dict.items():
                if '0' in name:
                    new_key = name.replace('0', 'normalize')
                if 'frozen_layers' in name:
                    new_key = name[16:]
                if 'head' in name:
                    new_key = name.replace('1.head.last_layer', 'fc')
                new_state_dict[new_key] = item
            state_dict_new = new_state_dict
    else:
        if args.bnNameCnt >= 0:
            for name, item in state_dict.items():
                if 'bn' in name:
                    assert 'bn_list' in name
                    state_dict_new[name.replace(
                        '.bn_list.{}'.format(args.bnNameCnt), '')] = item

        name_to_del = []
        for name, item in state_dict_new.items():
            if 'bn' in name and 'adv' in name:
                name_to_del.append(name)
            if 'bn_list' in name:
                name_to_del.append(name)
            if 'fc' in name:
                name_to_del.append(name)
        for name in np.unique(name_to_del):
            del state_dict_new[name]

        # deal with down sample layer
        keys = list(state_dict_new.keys())[:]
        name_to_del = []
        for name in keys:
            if 'downsample.conv' in name:
                state_dict_new[name.replace(
                    'downsample.conv', 'downsample.0')] = state_dict_new[name]
                name_to_del.append(name)
            if 'downsample.bn' in name:
                state_dict_new[name.replace(
                    'downsample.bn', 'downsample.1')] = state_dict_new[name]
                name_to_del.append(name)
        for name in np.unique(name_to_del):
            del state_dict_new[name]
        # state_dict_new['fc.weight'] = torch.zeros(num_classes, 512).cuda()
        # state_dict_new['fc.bias'] = torch.zeros(num_classes).cuda()
    return state_dict_new

def cvt_state_dict_AFF(state_dict, args):
    state_dict_new = copy.deepcopy(state_dict)

    if args.bnNameCnt >= 0:
        for name, item in state_dict.items():
            if 'bn' in name:
                assert 'bn_list' in name
                state_dict_new[name.replace(
                    '.bn_list.{}'.format(args.bnNameCnt), '')] = item

    name_to_del = []
    for name, item in state_dict_new.items():
        if 'bn' in name and 'adv' in name:
            name_to_del.append(name)
        if 'bn_list' in name:
            name_to_del.append(name)
        if 'fc' in name:
            name_to_del.append(name)
    for name in np.unique(name_to_del):
        del state_dict_new[name]

    # deal with down sample layer
    keys = list(state_dict_new.keys())[:]
    name_to_del = []
    for name in keys:
        if 'downsample.conv' in name:
            state_dict_new[name.replace(
                'downsample.conv', 'downsample.0')] = state_dict_new[name]
            name_to_del.append(name)
        if 'downsample.bn' in name:
            state_dict_new[name.replace(
                'downsample.bn', 'downsample.1')] = state_dict_new[name]
            name_to_del.append(name)
    for name in np.unique(name_to_del):
        del state_dict_new[name]

    state_dict_new['fc.weight'] = state_dict['fc.weight']
    state_dict_new['fc.bias'] = state_dict['fc.bias']
    return state_dict_new
    
# def get_model(opt, device, i):
def get_model(opt, device):
    # Choose the embedding network
    if opt.network == 'Resnet18':
        # model = resnet18()
        model = ResNet18()
        # model = ResNet50()
        # model = create_network()
        if opt.attack[0] == True:#/home/users/pxx/workplace/5Adversarial/4Meta_AT_ResNet/results/MAT_Cifar_single_test/Attack/best_checkpint_47_epoch.pkl
            path_checkpoint = r"/home/users/pxx/workplace/5Adversarial/9Rubost_AT/Bag-of-Tricks-for-AT-master/trained_models/cifar_model/model_best.pth"
        else:                  #/home/users/pxx/workplace/5Adversarial/MADS/Model_ResNet_Cifar_10/checkpint_181_epoch.pkl
            # path_checkpoint ="/home/users/pxx/workplace/5Adversarial/4Meta_AT_ResNet/results/1/best_MAT_checkpint_attack_{}.pkl".format(i)
            path_checkpoint = r"/home/users/pxx/workplace/2DeepEyes/CV-Baseline/06ResNet/ResNet/results/07-12_09-56/checkpoint_best.pkl"
        data_dir = r"/home/users/pxx/workplace/5Adversarial/MADS/MAD-C"     #  r"/home/users/pxx/workplace/5Adversarial/MADS/MAD-C"
    #/home/users/pxx/workplace/5Adversarial/Meta_Datasets/new/new_MNIST/test
    elif opt.network == 'EfficientNet':
        model = EfficientNet.from_name(model_name='efficientnet-b0', num_classes=200, image_size=224)
        if opt.attack[0] == True:
            path_checkpoint =r"/home/users/pxx/workplace/5Adversarial/4Meta_AT_ResNet/results/Model_EfficientNet_TinyImageNet/checkpint_best.pkl"
        else:
            path_checkpoint =r"/home/users/pxx/workplace/5Adversarial/4Meta_AT_ResNet/results/Model_EfficientNet_TinyImageNet/checkpint_best.pkl"
        data_dir = r"/home/users/pxx/workplace/5Adversarial/4Meta_AT_ResNet/data/TinyImageNet/val"
    elif opt.network == 'AlexNet':
        model = AlexNet()
        if opt.attack[0] == True:
            path_checkpoint =r"/home/users/pxx/workplace/5Adversarial/4Meta_AT_ResNet/results/AT_MNIST/checkpoint_best.pkl"
        else:
            # path_checkpoint ="/home/users/pxx/workplace/2DeepEyes/CV-Baseline/06ResNet/ResNet/results/07-09_05-16/checkpint_{}_epoch.pkl".format(i)
            path_checkpoint ="/home/users/pxx/workplace/5Adversarial/4Meta_AT_ResNet/results/07-11_16-34/best_checkpint_49_epoch.pkl"
        data_dir = r"/home/users/pxx/workplace/5Adversarial/4Meta_AT_ResNet/data/new_MNIST/test"
    else:
        print ("Cannot recognize the network type")
    model.to(device)
    checkpoint = torch.load(path_checkpoint)['state_dict']
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:]  # 去掉前缀（去掉前七个字符）
        new_state_dict[name] = v  # 新字典的key值对应的value为一一对应的值。
    model.load_state_dict(new_state_dict, strict=True)  # 重新加载这个模型
    # checkpoint = torch.load(path_checkpoint)
    # model.load_state_dict(checkpoint)
    # model.load_state_dict(checkpoint['state_dict'])
    # model.load_state_dict(checkpoint['model_state_dict'])
    # Choose the freezy network
    if opt.freezy_weight == True:
        if opt.network == 'Resnet18':
            for param in model.parameters():
                param.requires_grad = False
            in_channel = model.linear.in_features
            model.linear = torch.nn.Linear(in_channel,10)
        elif opt.network == 'EfficientNet':
            for param in model.parameters():
                param.requires_grad = False
            in_channel = model._fc.in_features
            model._fc = torch.nn.Linear(in_channel,200)
        elif opt.network == 'AlexNet':
            for param in model.parameters():
                param.requires_grad = False
            in_channel = model.fc8.in_features
            model.fc8 = torch.nn.Linear(in_channel,10)
        model.to(device)
    model = nn.DataParallel(model).cuda()

    return model, checkpoint, data_dir


def load_SSL_model(args, device):
    from model.resnet_ssl import resnet18
    if args.pre_dataset == 'CIFAR_10':
        if args.dataset == 'CIFAR_10':
            num_classes = 10
            # victim_path = os.path.join('src_ssl/victims', str(args.pre_dataset), str(args.victim), str(args.pre_dataset))    /SSL_cifar10_r18_cifar10_AdvCL
            victim_path = os.path.join('/home/users/pxx/workplace/5Adversarial/9_2Robust_SSL/RobustSSL_Benchmark-main/checkpoints', "SSL_cifar10_r18_cifar10_"+str(args.victim), str(args.mode))
        else:
            num_classes = 43
            victim_path = os.path.join('src_ssl/victims', str(args.pre_dataset), str(args.victim), str(args.dataset))
        # encoder_path = [Path(victim_path) / ckpt for ckpt in os.listdir(Path(victim_path)) if ckpt.startswith(str(args.mode))][0]
        encoder_path = [Path(victim_path) / ckpt for ckpt in os.listdir(Path(victim_path)) if ckpt.endswith('bestAT.pt')][0]
        if args.victim in ['AdvCL', 'AInfoNCE']:
            do_normalize = 0
        else:
            do_normalize = 1
        model = resnet18(num_classes=num_classes, do_normalize=do_normalize)
        checkpoint = torch.load(encoder_path, map_location="cpu")
        if 'state_dict_dual' in checkpoint:
            state_dict = checkpoint['state_dict_dual']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
        print('read checkpoint {}'.format(str(encoder_path)))
        data_dir = r"/home/users/pxx/workplace/Datasets/MADS/MAD-C-S"
        model.to(device)

    return model, checkpoint, data_dir

def load_victim(args, device):
    from model.resnet_ssl import resnet18
    if args.pre_dataset == 'CIFAR_10':
        num_classes = 10
        # victim_path = os.path.join('src_ssl/victims', 'CIFAR_10', str(args.victim))
        # encoder_path = [Path(victim_path) / ckpt for ckpt in os.listdir(Path(victim_path)) if ckpt.startswith("SLF")][0]
        if args.victim in ['AdvCL', 'A-InfoNCE']:
            do_normalize = 0
        else: 
            do_normalize = 1
        model = resnet18(num_classes=num_classes, do_normalize=do_normalize)
        # checkpoint = torch.load(encoder_path)
        checkpoint = torch.load(args.model)
        state_dict = checkpoint['state_dict']

        new_ckpt = dict()
        for k, value in state_dict.items():
            if k.startswith('backbone'):
                new_ckpt[k.replace('backbone.', '')] = value
            elif k.startswith('classifier'):
                new_ckpt[k.replace('classifier', 'fc')] = value
        if True:
            model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                                          bias=False)

        model.load_state_dict(new_ckpt, strict=False)   #strict=False

        model.fc = nn.Identity()
        model.maxpool = nn.Identity()
        for name, param in model.named_parameters():
            if name not in new_ckpt.keys():
                print('Warning: Missing {} when loading state dict.'.format(name))   

    return model.to(device)   

# victim model source zoo: https://github.com/vturrisi/solo-learn
def get_model_ACL(args, device):
    from model.resnet_ssl import resnet18
    if args.pre_dataset == 'CIFAR_10':
        if args.dataset == 'CIFAR_10':
            ####### set do_normalize=1 if your model needs to normalize the input, otherwise set do_normalize=0 ########
            if args.victim in ['AdvCL', 'A-InfoNCE']:
                do_normalize = 0
            else:
                do_normalize = 1
            num_classes=10
            model = resnet18(num_classes=num_classes, do_normalize=do_normalize)
            parameters = list(model.parameters())
            print('len of parameters: {}'.format(len(parameters)))
            victim_path = os.path.join('src_ssl/victims', str(args.pre_dataset), str(args.victim), str(args.dataset))
            encoder_path = [Path(victim_path) / ckpt for ckpt in os.listdir(Path(victim_path)) if ckpt.startswith(str(args.victim))][0]
            checkpoint = torch.load(encoder_path, map_location="cpu")
            if 'state_dict_dual' in checkpoint:
                state_dict = checkpoint['state_dict_dual']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            if args.mode in ['SLF', 'ALF']:
                ####### convert to single batchnorm batch ########
                print('convert to single batchnorm batch.')
                if args.victim in ['AdvCL', 'A-InfoNCE']:
                    state_dict= load_BN_checkpoint_AdvCL(args, state_dict)
                elif args.victim == 'DeACL':
                    state_dict = load_BN_checkpoint_DeACL(args, state_dict)
                elif args.victim in ['ACL','DynACL','DynACL++','DynACL-AIR','ACL_AIR','DynACL-AIR++', 'DynACL-RCS'] :
                    args.bnNameCnt = 1
                    state_dict = cvt_state_dict(state_dict, args, num_classes=num_classes)
                if args.init == "None":
                    state_dict['fc.weight'] = torch.zeros(num_classes, 512).to(device)
                    state_dict['fc.bias'] = torch.zeros(num_classes).to(device)
                model.load_state_dict(state_dict, strict=False)
                print('read checkpoint {}'.format(encoder_path))
            elif args.mode == 'eval':
                if args.dualBN:
                    args.bnNameCnt = 1
                    state_dict = cvt_state_dict_AFF(state_dict,args)
                model.load_state_dict(state_dict, strict=False)
                print('read checkpoint {}'.format(encoder_path))
            elif args.mode == 'AFF':
                if args.victim in ['AdvCL', 'A-InfoNCE']:
                    state_dict = load_BN_checkpoint_AdvCL(args, state_dict)
                elif args.victim == 'DeACL':
                    state_dict = load_BN_checkpoint_DeACL(args, state_dict)
                elif not args.dualBN and args.victim in ['ACL','DynACL','DynACL++','DynACL-AIR','ACL_AIR','DynACL-AIR++','DynACL-RCS'] :
                    args.bnNameCnt = 1
                    state_dict = cvt_state_dict(state_dict, args, num_classes=num_classes)
                if args.init == "None":
                    state_dict['fc.weight'] = torch.zeros(num_classes, 512).to(device)
                    state_dict['fc.bias'] = torch.zeros(num_classes).to(device)   
                model.load_state_dict(state_dict, strict=False)
                print('read checkpoint {}'.format(encoder_path))

            for name, param in model.named_parameters():
                if name not in state_dict.keys():
                    print('Warning: Missing {} when loading state dict.'.format(name))

        model.to(device)    

    return model

def get_dataset_train(opt):
    from data.attacked_dataset_train import ATTACKED_DATASET, FewShotDataloader
    # Choose the dataset
    dataset_train = ATTACKED_DATASET(opt, phase ='train')
    dataset_val = ATTACKED_DATASET(opt, phase='val')
    if opt.dataset == "TinyImageNet":
        num_class = len(TINYIMAGENET_CLASS_NAMES)
    else:
        num_class = len(CIFAR_CLASS_NAMES)
    dloader_train = FewShotDataloader(
        dataset=dataset_train,
        nKnovel=opt.train_query_way,
        nKbase=opt.train_shot_way,
        nExemplars=opt.train_shot, # num training examples per novel category
        nTestNovel=opt.train_query_way * opt.train_query * num_class, # num test examples for all the novel categories
        nTestBase=opt.train_shot_way * opt.train_shot * num_class, # num test examples for all the base categories
        batch_size=1,
        num_workers=32,
        epoch_size=opt.train_episode, # num of batches per epoch
        mix=True,
    )
    dloader_val = FewShotDataloader(
        dataset=dataset_val,
        nKnovel=opt.val_query_way,
        nKbase=opt.val_shot_way,
        nExemplars=opt.val_shot, # num training examples per novel category
        nTestNovel=opt.val_query_way * opt.val_query * num_class, # num test examples for all the novel categories
        nTestBase=opt.val_shot_way * opt.val_shot * num_class, # num test examples for all the base categories
        batch_size=1,
        num_workers=32,
        epoch_size=opt.val_episode, # num of batches per epoch
        mix=False,    # True, False
    )
    return (dloader_train, dloader_val)

def get_dataset_train_without_class(opt):
    from data.attacked_dataset_train_without_class import ATTACKED_DATASET, FewShotDataloader
    # Choose the dataset
    dataset_train = ATTACKED_DATASET(opt, phase ='train')
    dataset_val = ATTACKED_DATASET(opt, phase='val')
    if opt.dataset == "TinyImageNet":
        num_class = len(TINYIMAGENET_CLASS_NAMES)
    else:
        num_class = len(CIFAR_CLASS_NAMES)
    dloader_train = FewShotDataloader(
        dataset=dataset_train,
        nKnovel=opt.train_query_way,
        nKbase=opt.train_shot_way,
        nExemplars=opt.train_shot, # num training examples per novel category
        nTestNovel=opt.train_query_way * opt.train_query, # num test examples for all the novel categories
        nTestBase=opt.train_shot_way * opt.train_shot, # num test examples for all the base categories
        batch_size=opt.batch_size,
        num_workers=0,
        epoch_size=opt.train_episode, # num of batches per epoch
        mix=True,
    )
    dloader_val = FewShotDataloader(
        dataset=dataset_val,
        nKnovel=opt.val_query_way,
        nKbase=opt.val_shot_way,
        nExemplars=opt.val_shot, # num training examples per novel category
        nTestNovel=opt.val_query_way * opt.val_query, # num test examples for all the novel categories
        nTestBase=opt.val_shot_way * opt.val_shot, # num test examples for all the base categories
        batch_size=1,
        num_workers=16,
        epoch_size=opt.val_episode, # num of batches per epoch
        mix=False,    # True, False
    )
    return (dloader_train, dloader_val)

def get_dataset_test_single(opt, phase ='test', attack_num=0):   # train, val, test
    from data.attacked_dataset_test import ATTACKED_DATASET_TEST, FewShotDataloader_Test
    # Choose the dataset
    dataset_test = ATTACKED_DATASET_TEST(opt, phase)
    if opt.dataset == "TinyImageNet":
        num_class = len(TINYIMAGENET_CLASS_NAMES)
    else:
        num_class = len(CIFAR_CLASS_NAMES)
    dloader_test = FewShotDataloader_Test(
        dataset=dataset_test,
        attack_ID=attack_num,
        nExemplars=opt.test_shot, # num training examples per novel category
        nTestNovel=opt.test_query_way * opt.test_query * num_class, # num test examples for all the novel categories
        nTestBase=opt.test_shot_way * opt.test_shot * num_class, # num test examples for all the base categories
        batch_size=1,
        num_workers=16,
        epoch_size=1 * opt.test_episode, # num of batches per epoch
        mix=False,
    )
    return dloader_test

def get_dataset_test_all(opt, phase ='test'):
    from data.attacked_dataset_train import ATTACKED_DATASET, FewShotDataloader
    # Choose the dataset
    dataset_test = ATTACKED_DATASET(opt, phase)
    if opt.dataset == "TinyImageNet":
        num_class = len(TINYIMAGENET_CLASS_NAMES)
    else:
        num_class = len(CIFAR_CLASS_NAMES)
    dloader_test = FewShotDataloader(
        dataset=dataset_test,
        nKnovel=opt.test_query_way,
        nKbase=opt.test_shot_way,
        nExemplars=opt.test_shot, # num training examples per novel category
        nTestNovel=opt.test_query_way * opt.test_query * num_class, # num test examples for all the novel categories
        nTestBase=opt.test_shot_way * opt.test_shot * num_class, # num test examples for all the base categories
        batch_size=1,
        num_workers=0,
        epoch_size=1 * opt.test_episode, # num of batches per epoch
        mix=False,
    )
    return dloader_test


def get_loader(args):
    # setup data loader
    transform_train = transforms.Compose([
        transforms.Resize((args.resize, args.resize)),
        transforms.RandomCrop(args.resize if args.dataset == 'stl10' else 32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.resize, args.resize)),
        transforms.ToTensor(),
    ])

    if args.dataset == 'CIFAR_10':
        train_datasets = torchvision.datasets.CIFAR10(
            root=args.data, train=True, download=False, transform=transform_train)
        vali_datasets = torchvision.datasets.CIFAR10(
            root=args.data, train=False, download=False, transform=transform_test)
        testset = torchvision.datasets.CIFAR10(
            root=args.data, train=False, download=False, transform=transform_test)
        # train_datasets = CifarDataset(args.data + '/train', transform=transform_train)
        # vali_datasets = CifarDataset(args.data + '/val', transform=transform_test)
        # testset = CifarDataset(args.data + '/test', transform=transform_test)
        num_classes = 10
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        train_datasets = torchvision.datasets.CIFAR100(
            root=args.data, train=True, download=True, transform=transform_train)
        vali_datasets = torchvision.datasets.CIFAR100(
            root=args.data, train=True, download=True, transform=transform_test)
        testset = torchvision.datasets.CIFAR100(
            root=args.data, train=False, download=True, transform=transform_test)
        num_classes = 100
        args.num_classes = 100
    elif args.dataset == 'stl10':
        train_datasets = torchvision.datasets.STL10(
            root=args.data, split='train', transform=transform_train, download=True)
        vali_datasets = datasets.STL10(
            root=args.data, split='train', transform=transform_test, download=True)
        testset = datasets.STL10(
            root=args.data, split='test', transform=transform_test, download=True)
        num_classes = 10     
        args.num_classes = 10
    elif args.dataset == 'gtsrb':
        train_datasets = datasets.ImageFolder(args.data + '/train', transform=transform_train)
        vali_datasets = datasets.ImageFolder(args.data + '/val', transform=transform_test)
        testset = datasets.ImageFolder(args.data + '/test', transform=transform_test)
        num_classes = 43     
        args.num_classes = 43
    else:
        print("dataset {} is not supported".format(args.dataset))
        assert False

    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=args.batch_size, shuffle=True)
    vali_loader = torch.utils.data.DataLoader(vali_datasets, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    return train_loader, vali_loader, test_loader, num_classes, args