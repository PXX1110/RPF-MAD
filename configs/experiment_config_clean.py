import os
from typing import OrderedDict
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from model.alexnet import AlexNet
from model.efficientnet_pytorch.model import EfficientNet
from model.resnet_flc import resnet18
from model.resnet_s import ResNet18, ResNet50
from model.preactresnet import create_network
from pathlib import Path
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
            victim_path = os.path.join('src_ssl/victims', str(args.pre_dataset), str(args.victim), str(args.pre_dataset))
        else:
            num_classes = 43
            victim_path = os.path.join('src_ssl/victims', str(args.pre_dataset), str(args.victim), str(args.dataset))
        encoder_path = [Path(victim_path) / ckpt for ckpt in os.listdir(Path(victim_path)) if ckpt.startswith(str(args.mode))][0]
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


def get_dataset_train(opt):
    from data.attacked_dataset_train_with_clean import ATTACKED_DATASET, FewShotDataloader
    from data.cifar10_dataset import CifarDataset
    data_transforms = transforms.Compose([transforms.Resize((32, 32)),transforms.ToTensor()])
    # Choose the dataset
    adv_dataset_train = ATTACKED_DATASET(opt, phase ='train')
    adv_dataset_val = ATTACKED_DATASET(opt, phase='val')
    clean_dataset_train = CifarDataset(r'/home/users/pxx/workplace/Datasets/MADS/new_CIFAR_10_all',transform=data_transforms)
    clean_dataset_val = CifarDataset(r'/home/users/pxx/workplace/Datasets/MADS/new_CIFAR_10_all',transform=data_transforms)
    if opt.dataset == "TinyImageNet":
        num_class = len(TINYIMAGENET_CLASS_NAMES)
    else:
        num_class = len(CIFAR_CLASS_NAMES)
    dloader_train = FewShotDataloader(
        adv_dataset=adv_dataset_train,
        clean_dataset=clean_dataset_train,
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
        adv_dataset=adv_dataset_val,
        clean_dataset=clean_dataset_val,
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