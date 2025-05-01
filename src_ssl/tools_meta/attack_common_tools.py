# -*- coding: utf-8 -*-
"""
# @file name  : common_tools.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2020-06-23
# @brief      : 通用函数
"""
import os
import sys
sys.path.append(r'/home/users/pxx/workplace/5Adversarial/4Meta_AT_ResNet')
import time
import imageio
import numpy as np
import torch
import cv2
import torchvision.transforms as transforms
# from skimage import transform as trans
from PIL import Image   
from matplotlib import pyplot as plt
from art.estimators.classification.pytorch import PyTorchClassifier
from src_ssl.tools_meta.train_common_tools import ModelTrainer
from configs.experiment_config import CRITERTION
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def data_transform(opt, img_rgb, phase):
    
    train_transform_MNIST = transforms.Compose([transforms.Resize([32]), transforms.RandomHorizontalFlip(), lambda x: np.asarray(x), transforms.ToTensor()])
    train_transform_CIFAR_10 = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), lambda x: np.asarray(x), transforms.ToTensor()])
    # train_transform_TINIMAGENET = transforms.Compose([transforms.Resize([224]), transforms.RandomHorizontalFlip(), lambda x: np.asarray(x), transforms.ToTensor()])
    train_transform_TINIMAGENET = transforms.Compose([transforms.Resize([32]), transforms.RandomHorizontalFlip(), lambda x: np.asarray(x), transforms.ToTensor()])
   
    # test_transform = transforms.Compose([lambda x: np.asarray(x), transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.Resize([32]), lambda x: np.asarray(x), transforms.ToTensor()])
    # test_transform_TINIMAGENET = transforms.Compose([transforms.Resize([224]), lambda x: np.asarray(x), transforms.ToTensor()])
    test_transform_TINIMAGENET = transforms.Compose([transforms.Resize([32]), lambda x: np.asarray(x), transforms.ToTensor()])
    if phase == "train":
        if opt.dataset == 'MNIST':
            img_tensor = train_transform_MNIST(img_rgb)
        elif opt.dataset == 'CIFAR_10':
            img_tensor = train_transform_CIFAR_10(img_rgb)
        elif opt.dataset == 'TinyImageNet':
            img_tensor = train_transform_TINIMAGENET(img_rgb)   
    else:
        if opt.dataset == 'TinyImageNet':
            img_tensor = test_transform_TINIMAGENET(img_rgb)
        else:
            img_tensor = test_transform(img_rgb)

    return img_tensor

def my_mkdir(my_dir):
    if not os.path.isdir(my_dir):
        os.makedirs(my_dir)

def get_classifier(model, optimizer, input_shape, nb_classes):
    classifier = PyTorchClassifier(
            model=model,
            clip_values=(0.0, 1.0),
            loss=CRITERTION,
            optimizer=optimizer,
            input_shape=input_shape,
            nb_classes=nb_classes,
        )
    return classifier

def save_image(i, m, adv_examples, label_num, log_dir):
    # m=999
    if torch.is_tensor(adv_examples):
        adv_examples = adv_examples.cpu().detach().numpy()
    adv_examples = (np.squeeze(adv_examples*255)).transpose(1, 2, 0).astype(np.uint8)  
    o_dir = os.path.join(log_dir, "{}", label_num).format(i)
    my_mkdir(o_dir)
    img_name = label_num + '_' + str(m) + '.png'
    img_path = os.path.join(o_dir, img_name)
    imageio.imwrite(img_path, adv_examples)
    # now_time = datetime.now()
    # time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    # print(time_str)
    return adv_examples

def process_img(path_img, opt):
    # path --> img
    img_rgb = Image.open(path_img).convert('RGB')
    # img --> tensor
    phase = "test"
    img_tensor = data_transform(opt, img_rgb, phase)
    # img_rgb = img_rgb.resize((224, 224))
    img_tensor.unsqueeze_(0)        # chw --> bchw
    return img_tensor, img_rgb

def inference(img, label, model, cls_n, path_img, acc):

    with torch.no_grad():
        time_tic = time.time()
        outputs = model(img.to(device))
        time_toc = time.time()

    # 4/5 index to class names
    # 统计
    pred = torch.argmax(outputs, dim=1)
    label = label.to(device)
    acc += (pred ==label).sum()
    
    _, pred_int = torch.max(outputs.data, 1)
    _, top5_idx = torch.topk(outputs.data, 5, dim=1)

    pred_idx = int(pred_int.cpu().numpy())
    pred_str= cls_n[pred_idx]
    # print("img: {} is: {} ".format(os.path.basename(path_img), pred_str))
    # print("time consuming:{:.2f}s".format(time_toc - time_tic)) 
    return pred_str, top5_idx, acc

def show_image(img_rgb, pred_str, top5_idx, cls_n):

    plt.imshow(img_rgb)
    plt.title("predict:{}".format(pred_str))
    top5_num = top5_idx.cpu().numpy().squeeze()
    text_str = [cls_n[t] for t in top5_num]
    for idx in range(len(top5_num)):
        plt.text(0, idx*int((img_rgb.height/2)/(len(top5_num))), "top {}:{}".format(idx+1, text_str[idx]), bbox=dict(fc='yellow'))


def test_attack_samples(i, atk_key, atk, cls_n, model, data_dir, log_dir, opt):
    acc = 0
    adv_acc = 0
    n = 0
    for label_files in os.listdir(data_dir):
        label_file = os.path.join(data_dir, label_files)
        # fp = open(os.path.join(log_dir, label_files+".txt"), 'w')
        for path in os.listdir(label_file):
        #     '''
        #     #     this is hyly
        #     '''
        #     o_dir = os.path.join(log_dir, "{}", label_files).format(i)
        #     img_name = label_files + '_' + str(n) + '.png'
        #     n += 1
        #     fp.write(img_name)
        #     fp.write('\n')
        #     '''
            # path == os.listdir(label_file)[999]
            # 1/5 load img
            path_img = os.path.join(label_file, path)
            img_tensor, img_rgb = process_img(path_img, opt)
            label_num = np.array([int(label_files)]).astype(np.int64)
            label = torch.tensor(label_num)
            # 3/5 inference  tensor --> vector
            pred_str, top5_idx, acc = inference(img_tensor, label, model, cls_n, path_img, acc)
            # 4/5 attack
            adv_examples = ModelTrainer.attack(img_tensor, img_rgb, label_num, label, atk_key, atk)
            adv_images = save_image(i, n, adv_examples, label_files, log_dir)
            n += 1
            if len(adv_examples.shape) == 3:
                adv_examples = np.expand_dims(adv_examples, axis=0)
            if not torch.is_tensor(adv_examples):
                adv_examples = torch.tensor(adv_examples, dtype=torch.float32)
            adv_pred_str, adv_top5_idx, adv_acc  = inference(adv_examples, label, model, cls_n, path_img, adv_acc) 
            # # 5/5 visualization
            # plt.figure()
            # plt.subplot(1,2,1)
            # show_image(img_rgb, pred_str, top5_idx, cls_n)
            # plt.subplot(1,2,2)
            # show_image(Image.fromarray(adv_images).convert("RGB"), adv_pred_str, adv_top5_idx, cls_n)
            # plt.show()
            acc_avg = acc/n
            adv_acc_avg = adv_acc/n
            # if adv_acc_avg < acc_avg:
                # adv_images = save_image(i, n, adv_examples, label_files, log_dir)
            print('acc:{},adv_acc:{}'.format(acc_avg,adv_acc_avg))
            # '''
        # fp.close() 
    return acc_avg, adv_acc_avg 