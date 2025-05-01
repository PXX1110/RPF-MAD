# -*- coding: utf-8 -*-
"""
# @file name  : common_tools.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2020-06-23
# @brief      : 通用函数
"""
import sys,os
sys.path.append(r'/home/users/pxx/workplace/5Adversarial/4Meta_AT_ResNet/src_ssl/tools_meta')
import time
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
from tqdm import tqdm     
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable                             # 坑
from configs.attack_list_config import NOISE_PARAMS
from configs.experiment_config import CRITERTION
from perturbations import PERTURBATIONS
from early_stop import EarlyStopping
from copy import deepcopy
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()

def compute_intra_class_variance(embeddings, labels, num_classes):
    # 计算类内方差
    class_variances = []
    for label in range(num_classes):
        # 获取该类别的所有嵌入
        class_embeddings = embeddings[labels == label]
        # 计算该类别的均值
        class_mean = np.mean(class_embeddings, axis=0)
        # 计算该类别的方差
        variance = np.mean(np.linalg.norm(class_embeddings - class_mean, axis=1)**2)
        class_variances.append(variance)
    return np.mean(class_variances)

# Function to compute inter-class variance
def compute_inter_class_variance(embeddings, labels, num_classes):
    overall_mean = np.mean(embeddings, axis=0)
    class_means = [np.mean(embeddings[labels == label], axis=0) for label in range(num_classes)]
    inter_variance = np.sum([np.linalg.norm(class_mean - overall_mean)**2 for class_mean in class_means])
    return inter_variance / num_classes

def normalize_PGDAT(X):
    return (X - mu) / std

def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)

def check_dir(path):
    '''
    Create directory if it does not exist.
        path:           Path of directory.
    '''
    if not os.path.exists(path):
        os.mkdir(path)

def count_accuracy(logits, label):
    pred = torch.argmax(logits, dim=1).view(-1)
    label = label.view(-1)
    accuracy = 100 * pred.eq(label).float().mean()
    return accuracy

class Timer():
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / float(p)
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        # return x
        return '{}s'.format(x)

def log(log_file_path, string):
    '''
    Write one line of log into screen and file.
        log_file_path: Path of log file.
        string:        String to write in log file.
    '''
    with open(log_file_path, 'a+') as f:
        f.write(string + '\n')
        f.flush()
    print(string)
    
def get_resnet_18(path_state_dict, device, vis_model=False):
    """
    创建模型，加载参数
    :param path_state_dict:
    :return:
    """
    model = models.resnet18()
    if path_state_dict:
        pretrained_state_dict = torch.load(path_state_dict)
        model.load_state_dict(pretrained_state_dict)
    model.eval()

    if vis_model:
        from torchsummary import summary
        summary(model, input_size=(3, 224, 224), device="cpu")

    model.to(device)
    return model


def get_resnet_50(path_state_dict, device, vis_model=False):
    """
    创建模型，加载参数
    :param path_state_dict:
    :return:
    """
    model = models.resnet50()
    if path_state_dict:
        pretrained_state_dict = torch.load(path_state_dict)
        model.load_state_dict(pretrained_state_dict)
    model.eval()

    if vis_model:
        from torchsummary import summary
        summary(model, input_size=(3, 224, 224), device="cpu")

    model.to(device)
    return model

def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def KL(P,Q,mask=None):
    eps = 0.0000001
    d = (P+eps).log()-(Q+eps).log()
    d = P*d
    if mask !=None:
        d = d*mask
    return torch.sum(d)

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def CE(P,Q,mask=None):
    return KL(P,Q,mask)+KL(1-P,1-Q,mask)

def genetic_regularization(adv_results, ben_results, y, sample_weight=0, eps=0.0000001):
    (n, d) = adv_results.shape
    distance = 2.0
    tahn =  nn.Tanh()
    sample_weight = sample_weight.view(-1,n)
    sample_weight_matrix = (sample_weight+sample_weight.t())/32.0
    sample_weight_matrix = tahn(sample_weight_matrix)
    y  = y.view(-1,n)
    mask =1- (y==y.t()).float()
    mask[mask == 0] = -1
    distance = distance*mask*sample_weight_matrix
    adv_results_norm = torch.sqrt(torch.sum(adv_results ** 2, dim=1, keepdim=True))
    adv_results = adv_results / (adv_results_norm + eps)
    adv_results[adv_results != adv_results] = 0

    ben_results_norm = torch.sqrt(torch.sum(ben_results ** 2, dim=1, keepdim=True))
    ben_results = ben_results / (ben_results_norm + eps)
    ben_results[ben_results != ben_results] = 0

    model_similarity = torch.mm(adv_results, adv_results.transpose(0, 1))
    model_distance = 1 - model_similarity
    model_distance[range(n), range(n)] = 100000
    model_distance = model_distance - torch.min(model_distance, dim=1)[0].view(-1, 1)
    model_distance[range(n), range(n)] = 0
    model_distance = torch.clamp(model_distance, 0+eps, 2.0-eps)
    model_similarity = 1 - model_distance

    target_similarity = torch.mm(ben_results, ben_results.transpose(0, 1))
    target_distance = 1 - target_similarity
    target_distance[range(n), range(n)] = 100000
    p = torch.min(target_distance, dim=1)
    target_distance = target_distance - p[0].view(-1, 1)
    target_distance[range(n), range(n)] = 0
    target_distance = (1 - sample_weight_matrix) * target_distance + distance
    target_distance = torch.clamp(target_distance, 0+eps, 2.0-eps)
    target_similarity = 1 - target_distance

    model_similarity = (model_similarity + 1.0) / 2.0
    target_similarity = (target_similarity + 1.0) / 2.0

    model_similarity = model_similarity / torch.sum(model_similarity, dim=1, keepdim=True)
    target_similarity = target_similarity / torch.sum(target_similarity, dim=1, keepdim=True)
    loss = CE(target_similarity, model_similarity)
    return loss

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=10, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

def compute_adversarial_loss(model, clean_data, adv_data, labels, loss_f):
    # Forward pass
    adv_logits = model(adv_data)
    clean_logits = model(clean_data)
    # Compute the sample weights using clean logits
    sample_weight = torch.nn.CrossEntropyLoss(reduction='none')(clean_logits, labels)
    # Apply genetic regularization
    tp_loss = genetic_regularization(clean_logits, adv_logits, labels, sample_weight)
    # Standard adversarial loss
    ft_loss = loss_f(adv_logits, labels)
    # Compute final adversarial loss
    adv_loss = 20 * tp_loss + ft_loss
    return adv_loss, adv_logits

class ModelTrainer(object):
    
    @staticmethod
    def train(epoch, data_loader, model, loss_f, optimizer, lr_schedule, max_epoch, log_file_path, log_dir, opt):
        
        criterion_kl = nn.KLDivLoss(reduction='batchmean')  # Updated KLDivLoss to be more efficient  size_average=False
        phase = 'train'
        model.train()
        train_accuracies = []
        train_losses = []
        loss_rec = {"train": [], "valid": []}
        acc_rec = {"train": [], "valid": []}
        early_stopping = EarlyStopping(patience=20, verbose=True)

        for i, data in enumerate(tqdm(data_loader(epoch))):
            # Learning rate scheduling
            epoch_now = (epoch - 1) + (i + 1) / len(data_loader)
            lr = lr_schedule(epoch_now)
            optimizer.param_groups[0].update(lr=lr)

            # Data loading and reshaping
            data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in data]
            data_support = data_support.reshape([-1] + list(data_support.shape[-3:]))
            data_query = data_query.reshape([-1] + list(data_query.shape[-3:]))
            labels_support = labels_support.reshape(-1)
            labels_query = labels_query.reshape(-1)

            if opt.mixup:
                data_support_X, labels_support_a, labels_support_b, lam = mixup_data(data_support, labels_support, opt.mixup_alpha)
                data_support_X, labels_support_a, labels_support_b = map(Variable, (data_support_X, labels_support_a, labels_support_b))
                data_query_X, labels_query_a, labels_query_b, lam = mixup_data(data_query, labels_query, opt.mixup_alpha)
                data_query_X, labels_query_a, labels_query_b = map(Variable, (data_query_X, labels_query_a, labels_query_b))
                outputs_s = model(normalize_PGDAT(data_support_X))
                soft_labels_support = F.one_hot(labels_support, num_classes=outputs_s.size(1)).float()

                # Natural and Robust Loss
                loss_natural_s = mixup_criterion(loss_f, outputs_s, labels_support_a, labels_support_b, lam)
                loss_robust_s = criterion_kl(F.log_softmax(outputs_s, dim=1), F.softmax(soft_labels_support, dim=1)) / data_support.size(0)
                loss_s = loss_natural_s + loss_robust_s

                with torch.no_grad():
                    outputs_q = model(normalize_PGDAT(data_query_X))
                    soft_labels_query = F.one_hot(labels_query, num_classes=outputs_q.size(1)).float()
                    loss_natural_q = mixup_criterion(loss_f, outputs_q, labels_query_a, labels_query_b, lam)
                    loss_robust_q = criterion_kl(F.log_softmax(outputs_q, dim=1), F.softmax(soft_labels_query, dim=1)) / data_query.size(0)
                    loss_q = loss_natural_q + loss_robust_q

            else:
                # No mixup
                outputs_s = model(normalize_PGDAT(data_support))
                soft_labels_support = F.one_hot(labels_support, num_classes=outputs_s.size(1)).float()

                # Natural and Robust Loss
                loss_natural_s = loss_f(outputs_s, labels_support)
                loss_robust_s = criterion_kl(F.log_softmax(outputs_s, dim=1), F.softmax(soft_labels_support, dim=1)) / data_support.size(0)
                loss_s = loss_natural_s + loss_robust_s

                with torch.no_grad():
                    outputs_q = model(normalize_PGDAT(data_query))
                    soft_labels_query = F.one_hot(labels_query, num_classes=outputs_q.size(1)).float()
                    loss_natural_q = loss_f(outputs_q, labels_query)
                    loss_robust_q = criterion_kl(F.log_softmax(outputs_q, dim=1), F.softmax(soft_labels_query, dim=1)) / data_query.size(0)
                    loss_q = loss_natural_q + loss_robust_q

            # Combined loss
            loss = loss_s * opt.lamda + loss_q * (1 - opt.lamda)

            # Optimizer steps
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()

            # Accuracy and loss logging
            acc = count_accuracy(outputs_q, labels_query)
            train_losses.append(loss.item())
            train_accuracies.append(acc.item())

            if i % 1 == 0:
                train_loss_avg = np.mean(train_losses)
                train_acc_avg = np.mean(np.array(train_accuracies))

            if not opt.earlystop_train:
                log(log_file_path, 'Train Epoch: [{:0>3}/{:0>3}] \t Batch: [{:0>3}/{:0>3}] \t Loss: {:.4f} \t Accuracy: {:.2f} %'
                    .format(epoch, max_epoch, i, len(data_loader), loss.item(), train_acc_avg))
            else:
                early_stopping(acc.item(), epoch, max_epoch, phase, 0, loss.item(), model, optimizer, 
                            loss_rec, acc_rec, log_dir, log_file_path, i, len(data_loader), 0)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        return train_loss_avg, train_acc_avg, lr, optimizer, model


    @staticmethod
    def valid(epoch, data_loader, model, loss_f, optimizer, max_epoch, log_file_path, log_dir, opt):
        
        criterion_kl = nn.KLDivLoss(size_average=False)
        phase='val'
        test_model = deepcopy(model)  # Copy the model for testing
        test_model.train()  # Set model in training mode to fine-tune in validation loop

        val_accuracies = []
        val_losses = []
        loss_rec = {"train": [], "valid": []}
        acc_rec = {"train": [], "valid": []}
        early_stopping = EarlyStopping(patience=20, verbose=True)

        for i, data in enumerate(tqdm(data_loader(epoch))):
            # Reshape data
            data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in data]
            data_support = data_support.reshape([-1] + list(data_support.shape[-3:]))
            data_query = data_query.reshape([-1] + list(data_query.shape[-3:]))
            labels_support = labels_support.reshape(-1)
            labels_query = labels_query.reshape(-1)
            
            # Fine-tuning the model for several steps on data_support
            for j in range(10):  # Fine-tune for a fixed number of steps
                optimizer.zero_grad()
                outputs_s = test_model(normalize_PGDAT(data_support))  # Normalize data
                loss_s = loss_f(outputs_s, labels_support)  # Compute the loss
                loss_s.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
                optimizer.step()

            # Set the model to evaluation mode for querying
            test_model.eval()
            with torch.no_grad():
                outputs_q = test_model(normalize_PGDAT(data_query))
                loss_q = loss_f(outputs_q, labels_query)  # Compute loss for validation query set
            
            # Compute accuracy
            acc = count_accuracy(outputs_q, labels_query)
            val_losses.append(loss_q.item())
            val_accuracies.append(acc.item())

            val_acc_avg = np.mean(val_accuracies)
            val_acc_ci95 = 1.96 * np.std(np.array(val_accuracies)) / np.sqrt(len(val_accuracies))
            val_loss_avg = np.mean(val_losses)  


            # Logging
            if not opt.earlystop_val:
                log(log_file_path, f'Validation Epoch: [{epoch:0>3}/{max_epoch:0>3}] \t Batch: [{i:0>3}/{len(data_loader):0>3}] \t Loss: {loss_q.item():.4f} \t Accuracy: {val_acc_avg:.2f} % ({val_acc_ci95:.2f} %)')
            else:
                early_stopping(acc.item(), epoch, max_epoch, phase, val_acc_ci95, loss_q.item(), model, optimizer,
                            loss_rec, acc_rec, log_dir, log_file_path, i, len(data_loader), 0)

            # Early stopping
            if early_stopping.early_stop:
                print("Early stopping")
                break

        return val_loss_avg, val_acc_avg
    
    @staticmethod
    def test(data_loader, model, loss_f, optimizer, log_file_path, log_dir, attack_num, opt):

        criterion_kl = nn.KLDivLoss(size_average=False)
        phase='test'
        test_model = deepcopy(model)  # Copy the model for testing
        test_model.train()  # Set model in training mode to fine-tune in validation loop

        test_accuracies = []
        test_losses = []
        loss_rec = {"train": [], "valid": []}
        acc_rec = {"train": [], "valid": []}
        best_acc, best_episode = 0.0, 0
        epoch, max_epoch = 0, 0
        early_stopping = EarlyStopping(patience=50, verbose=True)

        for i, data in enumerate(tqdm(data_loader(epoch))):
            # Reshape data
            data_support, labels_support, data_query, labels_query = [x.cuda() for x in data]   
            data_support = data_support.reshape([-1] + list(data_support.shape[-3:]))
            data_query = data_query.reshape([-1] + list(data_query.shape[-3:]))
            labels_support = labels_support.reshape(-1)
            labels_query = labels_query.reshape(-1)
            
            # Fine-tuning the model for several steps on data_support
            for j in range(10):  # Fine-tune for a fixed number of steps
                optimizer.zero_grad()
                outputs_s = test_model(normalize_PGDAT(data_support))  # Normalize data
                loss_s = loss_f(outputs_s, labels_support)  # Compute the loss
                loss_s.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
                optimizer.step()

            # Set the model to evaluation mode for querying
            test_model.eval()
            with torch.no_grad():
                outputs_q = test_model(normalize_PGDAT(data_query))
                loss_q = loss_f(outputs_q, labels_query)  # Compute loss for validation query set
            
            # Compute accuracy
            acc = count_accuracy(outputs_q, labels_query)
            test_losses.append(loss_q.item())
            test_accuracies.append(acc.item())

            test_acc_ci95 = 1.96 * np.std(test_accuracies) / np.sqrt(len(test_accuracies))

            # Early stopping mechanism
            early_stopping(acc.item(), epoch, max_epoch, phase, test_acc_ci95, loss_q.item(), test_model, optimizer, 
                           loss_rec, acc_rec, log_dir, log_file_path, i, len(data_loader), attack_num)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        test_acc_avg = np.mean(test_accuracies)
        test_loss_avg = np.mean(test_losses)

        return test_acc_avg, test_loss_avg
    
    @staticmethod
    def train_ssl(epoch, data_loader, model, loss_f, optimizer, lr_schedule, max_epoch, log_file_path, log_dir, opt):
        
        criterion_kl = nn.KLDivLoss(size_average=False)
        phase='train'
        model.train()

        train_accuracies = []
        train_losses = []
        loss_rec = {"train": [], "valid": []}
        acc_rec = {"train": [], "valid": []}
        early_stopping = EarlyStopping(patience=20, verbose=True)

        for i, data in enumerate(tqdm(data_loader(epoch))):

            epoch_now = (epoch-1) + (i + 1) / len(data_loader)
            lr = lr_schedule(epoch_now)
            optimizer.param_groups[0].update(lr=lr)

            data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in data]
            data_support = data_support.reshape([-1] + list(data_support.shape[-3:]))
            data_query = data_query.reshape([-1] + list(data_query.shape[-3:]))
            labels_support = labels_support.reshape(-1)
            labels_query = labels_query.reshape(-1)

            if opt.mixup:
                data_support_X, labels_support_a, labels_support_b, lam = mixup_data(data_support, labels_support, opt.mixup_alpha)
                data_support_X, labels_support_a, labels_support_b = map(Variable, (data_support_X, labels_support_a, labels_support_b))
                data_query_X, labels_query_a, labels_query_b, lam = mixup_data(data_query, labels_query, opt.mixup_alpha)
                data_query_X, labels_query_a, labels_query_b = map(Variable, (data_query_X, labels_query_a, labels_query_b))
                
                outputs_s = model(data_support_X)
                loss_natural_s = mixup_criterion(loss_f, outputs_s, labels_support_a, labels_support_b, lam)

                if opt.victim not in ["ACL"]:
                    soft_lables_support = F.one_hot(labels_support, num_classes=outputs_s.size(1)).float() 
                    loss_robust = (1.0 / data_support.size(0)) * criterion_kl(F.log_softmax(outputs_s, dim=1),
                                                        F.softmax(soft_lables_support, dim=1))
                    loss_s = loss_natural_s + loss_robust
                else:
                    loss_s = loss_natural_s

                with torch.no_grad():
                    outputs_q = model(data_query_X)
                    loss_natural_q = mixup_criterion(loss_f, outputs_q, labels_query_a, labels_query_b, lam)

                    if opt.victim not in ["ACL"]:
                        soft_lables_query = F.one_hot(labels_query, num_classes=outputs_q.size(1)).float()
                        loss_robust = (1.0 / data_query.size(0)) * criterion_kl(F.log_softmax(outputs_q, dim=1),
                                                        F.softmax(soft_lables_query, dim=1))
                        loss_q = loss_natural_q + loss_robust
                    else:
                        loss_q = loss_natural_q

            else:
                outputs_s = model(data_support)
                loss_natural_s = loss_f(outputs_s, labels_support)

                if opt.victim not in ["ACL"]:
                    soft_lables_support = F.one_hot(labels_support, num_classes=outputs_s.size(1)).float()  
                    loss_robust = (1.0 / data_support.size(0)) * criterion_kl(F.log_softmax(outputs_s, dim=1),
                                                        F.softmax(soft_lables_support, dim=1))
                    loss_s = loss_natural_s + loss_robust
                else:
                    loss_s = loss_natural_s

                with torch.no_grad():
                    outputs_q = model(data_query)
                    loss_natural_q = loss_f(outputs_q, labels_query)

                    if opt.victim not in ["ACL"]:
                        soft_lables_query = F.one_hot(labels_query, num_classes=outputs_q.size(1)).float()
                        loss_robust = (1.0 / data_query.size(0)) * criterion_kl(F.log_softmax(outputs_q, dim=1),
                                                        F.softmax(soft_lables_query, dim=1))
                        loss_q = loss_natural_q + loss_robust
                    else:
                        loss_q = loss_natural_q

            loss = loss_s * opt.lamda + loss_q * (1 - opt.lamda)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()

            acc = count_accuracy(outputs_q, labels_query)
            train_losses.append(loss.item())
            train_accuracies.append(acc.item())
            train_acc_ci95 = 0

            if (i % 1 == 0):
                train_loss_avg = np.mean(train_losses)
                train_acc_avg = np.mean(train_accuracies)

            if not opt.earlystop_train:
                log(log_file_path, 'Train Epoch: [{:0>3}/{:0>3}] \t Batch: [{:0>3}/{:0>3}] \t Loss: {:.4f} \t Accuracy: {:.2f} % ({:.2f} %)'
                    .format(epoch, max_epoch, i, len(data_loader), loss.item(), train_acc_avg, acc))
            else:
                early_stopping(acc.item(), epoch, max_epoch, phase, train_acc_ci95, loss.item(), model, optimizer, 
                           loss_rec, acc_rec, log_dir, log_file_path, i, len(data_loader), 0)
                
            if early_stopping.early_stop:
                print("Early stopping")
                break

        return train_loss_avg, train_acc_avg, lr, optimizer, model

    @staticmethod
    def valid_ssl(epoch, data_loader, model, loss_f, optimizer, max_epoch, log_file_path, log_dir, opt):
        
        criterion_kl = nn.KLDivLoss(size_average=False)
        phase='val'
        test_model = deepcopy(model)  # Copy the model for testing
        test_model.train()  # Set model in training mode to fine-tune in validation loop

        val_accuracies = []
        val_losses = []
        loss_rec = {"train": [], "valid": []}
        acc_rec = {"train": [], "valid": []}
        early_stopping = EarlyStopping(patience=20, verbose=True)

        for i, data in enumerate(tqdm(data_loader(epoch))):
            # Reshape data
            data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in data]
            data_support = data_support.reshape([-1] + list(data_support.shape[-3:]))
            data_query = data_query.reshape([-1] + list(data_query.shape[-3:]))
            labels_support = labels_support.reshape(-1)
            labels_query = labels_query.reshape(-1)

            # Fine-tuning the model for several steps on data_support
            for j in range(10): # Fine-tune for a fixed number of steps
                optimizer.zero_grad()
                outputs_s = test_model(data_support)
                loss_s = loss_f(outputs_s, labels_support) # Compute the loss
                loss_s.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
                optimizer.step()

            # Set the model to evaluation mode for querying
            test_model.eval()
            with torch.no_grad():
                outputs_q = test_model(data_query)
                loss_q = loss_f(outputs_q, labels_query)
            
            # Compute accuracy
            acc = count_accuracy(outputs_q, labels_query)
            val_accuracies.append(acc.item())
            val_losses.append(loss_q.item())
            val_acc_avg = np.mean(val_accuracies)
            val_acc_ci95 = 1.96 * np.std(np.array(val_accuracies)) / np.sqrt(len(val_accuracies))
            val_loss_avg = np.mean(val_losses)

            # Logging
            if not opt.earlystop_val:
                log(log_file_path, 'Validation Epoch: [{:0>3}/{:0>3}] \t Batch: [{:0>3}/{:0>3}] \t Loss: {:.4f} \t Accuracy: {:.2f} % ({:.2f} %)'
                    .format(epoch, max_epoch, i, len(data_loader), loss_q.item(), val_acc_avg, val_acc_ci95))
            else:
                early_stopping(acc.item(), epoch, max_epoch, phase, val_acc_ci95, loss_q.item(), model, optimizer, 
                           loss_rec, acc_rec, log_dir, log_file_path, i, len(data_loader), 0)
            
            # Early stopping
            if early_stopping.early_stop:
                print("Early stopping")
                break

        return val_loss_avg, val_acc_avg
    
    @staticmethod
    def test_ssl(data_loader, model, loss_f, optimizer, log_file_path, log_dir, attack_num, opt):
        
        criterion_kl = nn.KLDivLoss(size_average=False)
        phase='test'
        test_model = deepcopy(model)  # Copy the model for testing
        test_model.train()  # Set model in training mode to fine-tune in validation loop

        test_accuracies = []
        test_losses = []
        loss_rec = {"train": [], "valid": []}
        acc_rec = {"train": [], "valid": []}
        best_acc, best_episode = 0.0, 0
        epoch, max_epoch = 0, 0
        early_stopping = EarlyStopping(patience=50, verbose=True)

        for i, data in enumerate(tqdm(data_loader(epoch))):
            # Reshape data
            data_support, labels_support, data_query, labels_query = [x.cuda() for x in data]   
            data_support = data_support.reshape([-1] + list(data_support.shape[-3:]))
            data_query = data_query.reshape([-1] + list(data_query.shape[-3:]))
            labels_support = labels_support.reshape(-1)
            labels_query = labels_query.reshape(-1)
            
            # Fine-tuning the model for several steps on data_support
            for j in range(10):  # Fine-tune for a fixed number of steps
                optimizer.zero_grad()
                outputs_s = test_model(data_support) # Normalize data
                loss_s = loss_f(outputs_s, labels_support)  # Compute the loss
                loss_s.backward()
                # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
                optimizer.step()

            # Set the model to evaluation mode for querying
            test_model.eval()
            with torch.no_grad():
                outputs_q = test_model(data_query)
                loss_q = loss_f(outputs_q, labels_query)  # Compute loss for validation query set
            
            # Compute accuracy
            acc = count_accuracy(outputs_q, labels_query)
            test_losses.append(loss_q.item())
            test_accuracies.append(acc.item())

            test_acc_ci95 = 1.96 * np.std(test_accuracies) / np.sqrt(len(test_accuracies))

            # Early stopping mechanism
            early_stopping(acc.item(), epoch, max_epoch, phase, test_acc_ci95, loss_q.item(), test_model, optimizer, 
                           loss_rec, acc_rec, log_dir, log_file_path, i, len(data_loader), attack_num)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        test_acc_avg = np.mean(test_accuracies)
        test_loss_avg = np.mean(test_losses)

        return test_acc_avg, test_loss_avg

    @staticmethod
    def train_ssl_clean(epoch, data_loader, model, loss_f, optimizer, lr_schedule, max_epoch, log_file_path, log_dir, opt):
        
        criterion_kl = nn.KLDivLoss(size_average=False)
        phase='train'
        model.train()

        train_accuracies = []
        train_losses = []
        loss_rec = {"train": [], "valid": []}
        acc_rec = {"train": [], "valid": []}
        early_stopping = EarlyStopping(patience=20, verbose=True)

        for i, data in enumerate(tqdm(data_loader(epoch))):

            epoch_now = (epoch-1) + (i + 1) / len(data_loader)
            lr = lr_schedule(epoch_now)
            optimizer.param_groups[0].update(lr=lr)

            data_support, labels_support, clean_support, data_query, labels_query, clean_query, _, _ = [x.cuda() for x in data]
            data_support = data_support.reshape([-1] + list(data_support.shape[-3:]))
            data_query = data_query.reshape([-1] + list(data_query.shape[-3:]))
            clean_support = clean_support.reshape([-1] + list(clean_support.shape[-3:]))
            clean_query = clean_query.reshape([-1] + list(clean_query.shape[-3:]))
            labels_support = labels_support.reshape(-1)
            labels_query = labels_query.reshape(-1)

            optimizer.zero_grad()
            if opt.mixup:

                data_support_X, labels_support_a, labels_support_b, lam = mixup_data(data_support, labels_support, opt.mixup_alpha)
                data_support_X, labels_support_a, labels_support_b = map(Variable, (data_support_X, labels_support_a, labels_support_b))
                data_query_X, labels_query_a, labels_query_b, lam = mixup_data(data_query, labels_query, opt.mixup_alpha)
                data_query_X, labels_query_a, labels_query_b = map(Variable, (data_query_X, labels_query_a, labels_query_b))

                outputs_s = model(data_support_X)
                soft_lables_support = F.one_hot(labels_support, num_classes=outputs_s.size(1)).float()
                # outputs_s = model(data_support)    
                loss_natural_s = mixup_criterion(loss_f, outputs_s, labels_support_a, labels_support_b, lam)
                loss_robust = (1.0 / data_support.size(0)) * criterion_kl(F.log_softmax(outputs_s, dim=1),
                                                    F.softmax(soft_lables_support, dim=1))
                loss_s = loss_natural_s + loss_robust

                with torch.no_grad():
                    outputs_q = model(data_query_X)
                    soft_lables_query = F.one_hot(labels_query, num_classes=outputs_q.size(1)).float()
                    loss_natural_q = mixup_criterion(loss_f, outputs_q, labels_query_a, labels_query_b, lam)
                    loss_robust = (1.0 / data_query.size(0)) * criterion_kl(F.log_softmax(outputs_q, dim=1),
                                                    F.softmax(soft_lables_query, dim=1))
                    # optimizer.zero_grad()
                    loss_q = loss_natural_q + loss_robust

            else:
                support_loss, support_logits = compute_adversarial_loss(model, clean_support, data_support, labels_support, loss_f)
                query_loss, query_logits = compute_adversarial_loss(model, clean_query, data_query, labels_query, loss_f)
                # Final loss by summing both losses
                loss = support_loss + query_loss

            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            # 统计
            acc = count_accuracy(query_logits, labels_query)
            train_losses.append(loss.item())
            train_accuracies.append(acc.item())
            train_acc_ci95 = 0

            if (i % 1 == 0):
                train_loss_avg = np.mean(train_losses)
                train_acc_avg = np.mean(np.array(train_accuracies))
            if not opt.earlystop_train:
                log(log_file_path, 'Train Epoch: [{:0>3}/{:0>3}] \t Batch: [{:0>3}/{:0>3}] \t Loss: {:.4f} \t Accuracy: {:.2f} % ({:.2f} %)'
                    .format(epoch, max_epoch, i, len(data_loader), loss.item(), train_acc_avg, acc))
            else:
                early_stopping(acc.item(), epoch, max_epoch, phase, train_acc_ci95, loss.item(), model, optimizer, 
                           loss_rec, acc_rec, log_dir, log_file_path, i, len(data_loader), 0)
                
            if early_stopping.early_stop:
                print("Early stopping")
                break

        return train_loss_avg, train_acc_avg, lr, optimizer, model

    @staticmethod
    def valid_ssl_clean(epoch, data_loader, model, loss_f, optimizer, max_epoch, log_file_path, log_dir, opt):
        
        criterion_kl = nn.KLDivLoss(size_average=False)
        phase='val'
        test_model = deepcopy(model)  # Copy the model for testing
        test_model.train()  # Set model in training mode to fine-tune in validation loop

        val_accuracies = []
        val_losses = []
        loss_rec = {"train": [], "valid": []}
        acc_rec = {"train": [], "valid": []}
        early_stopping = EarlyStopping(patience=20, verbose=True)

        for i, data in enumerate(tqdm(data_loader(epoch))):

            data_support, labels_support, clean_support, data_query, labels_query, clean_query, _, _ = [x.cuda() for x in data]
            data_support = data_support.reshape([-1] + list(data_support.shape[-3:]))
            data_query = data_query.reshape([-1] + list(data_query.shape[-3:]))
            clean_support = clean_support.reshape([-1] + list(clean_support.shape[-3:]))
            clean_query = clean_query.reshape([-1] + list(clean_query.shape[-3:]))
            labels_support = labels_support.reshape(-1)
            labels_query = labels_query.reshape(-1)

            for j in range(10):
                optimizer.zero_grad() 
                loss_s, _ = compute_adversarial_loss(test_model, clean_support, data_support, labels_support, loss_f)
                # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping 
                loss_s.backward()
                optimizer.step()

            test_model.eval()
            with torch.no_grad():
                outputs_q = test_model(data_query)
                loss_q = loss_f(outputs_q, labels_query)

            # 统计
            acc = count_accuracy(outputs_q, labels_query)
            val_accuracies.append(acc.item())
            val_losses.append(loss_q.item())
            val_acc_avg = np.mean(np.array(val_accuracies))
            val_acc_ci95 = 1.96 * np.std(np.array(val_accuracies)) / np.sqrt(opt.val_episode)
            val_loss_avg = np.mean(np.array(val_losses))   

            if not opt.earlystop_val:
                log(log_file_path, 'Validation Epoch: [{:0>3}/{:0>3}] \t Batch: [{:0>3}/{:0>3}] \t Loss: {:.4f} \t Accuracy: {:.2f} % ({:.2f} %)'
                    .format(epoch, max_epoch, i, len(data_loader), loss_q.item(), val_acc_avg, val_acc_ci95))
            else:
                early_stopping(acc.item(), epoch, max_epoch, phase, val_acc_ci95, loss_q.item(), model, optimizer, 
                           loss_rec, acc_rec, log_dir, log_file_path, i, len(data_loader), 0)
                
            if early_stopping.early_stop:
                print("Early stopping")
                break

        return val_loss_avg, val_acc_avg
    
    @staticmethod
    def test_ssl_clean(data_loader, model, loss_f, optimizer, log_file_path, log_dir, attack_num, opt):
        phase='test'
        test_accuracies = []
        test_losses = []
        loss_rec = {"train": [], "valid": []}
        acc_rec = {"train": [], "valid": []}
        best_acc, best_episode = 0.0, 0
        epoch, max_epoch = 0, 0
        early_stopping = EarlyStopping(patience=50, verbose=True)
        timer = time.time()

        for i, data in enumerate(tqdm(data_loader(epoch=0))):
            # data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in data]  # all_test
            data_support, labels_support, data_query, labels_query = [x.cuda() for x in data]          # single_test
            data_support = data_support.reshape([-1] + list(data_support.shape[-3:]))
            data_query = data_query.reshape([-1] + list(data_query.shape[-3:]))
            labels_support = labels_support.reshape(-1)
            labels_query = labels_query.reshape(-1)
            test_model = deepcopy(model)
            for j in range(10):
                outputs_s = test_model(data_support)    
                loss_s = loss_f(outputs_s, labels_support)
                optimizer.zero_grad()   
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping 
                loss_s.backward()
                optimizer.step()
                j += 1
                if j >= 10:
                    break
            test_model.eval()
            with torch.no_grad():
                outputs_q = test_model(data_query)
                loss_q = loss_f(outputs_q, labels_query)
            loss = loss_q
            # 统计
            acc = count_accuracy(outputs_q, labels_query)
            test_accuracies.append(acc.item())
            test_losses.append(loss.item())
            # test_acc_avg = np.mean(np.array(test_accuracies))
            test_acc_ci95 = 1.96 * np.std(np.array(test_accuracies)) / np.sqrt(opt.test_episode)
            early_stopping(acc.item(), epoch, max_epoch, phase, test_acc_ci95, loss.item(), model, optimizer, 
                        loss_rec, acc_rec, log_dir, log_file_path, i, len(data_loader), attack_num)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        return acc.item(), loss.item()

    @staticmethod
    # def train_ssl_multi(epoch, data_loader, model, loss_f, e_optimizer, f_optimizer, max_epoch, log_file_path, log_dir, opt):
    def train_ssl_multi(epoch, data_loader, model, loss_f, e_optimizer, f_optimizer, lr_schedule_e, lr_schedule_f, max_epoch, log_file_path, log_dir, opt):    
        
        criterion_kl = nn.KLDivLoss(size_average=False)
        phase='train'
        model.train()

        train_accuracies = []
        train_losses = []
        loss_rec = {"train": [], "valid": []}
        acc_rec = {"train": [], "valid": []}
        early_stopping = EarlyStopping(patience=20, verbose=True)

        for i, data in enumerate(tqdm(data_loader(epoch))):

            epoch_now = (epoch-1) + (i + 1) / len(data_loader)
            lr_e = lr_schedule_e(epoch_now)
            e_optimizer.param_groups[0].update(lr=lr_e)
            lr_f = lr_schedule_f(epoch_now)
            f_optimizer.param_groups[0].update(lr=lr_f)
            e_optimizer.zero_grad()
            f_optimizer.zero_grad()

            data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in data]
            data_support = data_support.reshape([-1] + list(data_support.shape[-3:]))
            data_query = data_query.reshape([-1] + list(data_query.shape[-3:]))
            labels_support = labels_support.reshape(-1)
            labels_query = labels_query.reshape(-1)
            
            if opt.mixup:
                data_support_X, labels_support_a, labels_support_b, lam = mixup_data(data_support, labels_support, opt.mixup_alpha)
                data_support_X, labels_support_a, labels_support_b = map(Variable, (data_support_X, labels_support_a, labels_support_b))
                data_query_X, labels_query_a, labels_query_b, lam = mixup_data(data_query, labels_query, opt.mixup_alpha)
                data_query_X, labels_query_a, labels_query_b = map(Variable, (data_query_X, labels_query_a, labels_query_b))
                outputs_s = model(data_support_X)
                loss_natural_s = mixup_criterion(loss_f, outputs_s, labels_support_a, labels_support_b, lam)
                if opt.victim not in ["ACL"]:
                    soft_lables_support = F.one_hot(labels_support, num_classes=outputs_s.size(1)).float()
                    # outputs_s = model(data_support)    
                    loss_robust = (1.0 / data_support.size(0)) * criterion_kl(F.log_softmax(outputs_s, dim=1),
                                                        F.softmax(soft_lables_support, dim=1))
                    loss_s = loss_natural_s + loss_robust
                else:
                    loss_s = loss_natural_s
                with torch.no_grad():
                    outputs_q = model(data_query_X)
                    loss_natural_q = mixup_criterion(loss_f, outputs_q, labels_query_a, labels_query_b, lam)
                    if opt.victim not in ["ACL"]:
                        soft_lables_query = F.one_hot(labels_query, num_classes=outputs_q.size(1)).float()
                        loss_robust = (1.0 / data_query.size(0)) * criterion_kl(F.log_softmax(outputs_q, dim=1),
                                                        F.softmax(soft_lables_query, dim=1))
                        # optimizer.zero_grad()
                        loss_q = loss_natural_q + loss_robust
                    else:
                        loss_q = loss_natural_q
            else:
                outputs_s = model(data_support)
                loss_natural_s = loss_f(outputs_s, labels_support)

                # if opt.victim not in ["ACL"]:
                #     soft_lables_support = F.one_hot(labels_support, num_classes=outputs_s.size(1)).float()
                #     # outputs_s = model(data_support)    
                #     loss_robust = (1.0 / data_support.size(0)) * criterion_kl(F.log_softmax(outputs_s, dim=1),
                #                                         F.softmax(soft_lables_support, dim=1))
                #     loss_s = loss_natural_s + loss_robust
                # else:
                loss_s = loss_natural_s

                with torch.no_grad():
                    outputs_q = model(data_query)
                    loss_natural_q = loss_f(outputs_q, labels_query)

                    # if opt.victim not in ["ACL"]:
                    #     soft_lables_query = F.one_hot(labels_query, num_classes=outputs_q.size(1)).float()
                    #     loss_robust = (1.0 / data_query.size(0)) * criterion_kl(F.log_softmax(outputs_q, dim=1),
                    #                                     F.softmax(soft_lables_query, dim=1))
                    #     loss_q = loss_natural_q + loss_robust
                    # else:
                    loss_q = loss_natural_q

            # loss = loss_q
            loss = loss_s * opt.lamda + loss_q * (1 - opt.lamda)
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            e_optimizer.step()
            f_optimizer.step()
            # 统计
            acc = count_accuracy(outputs_q, labels_query)
            train_losses.append(loss.item())
            train_accuracies.append(acc.item())
            train_acc_ci95 = 0
            if (i % 1 == 0):
                train_loss_avg = np.mean(train_losses)
                train_acc_avg = np.mean(np.array(train_accuracies))

            if not opt.earlystop_train:
                log(log_file_path, 'Train Epoch: [{:0>3}/{:0>3}] \t Batch: [{:0>3}/{:0>3}] \t Loss: {:.4f} \t Accuracy: {:.2f} % ({:.2f} %)'
                    .format(epoch, max_epoch, i, len(data_loader), loss.item(), train_acc_avg, acc))
            else:
                early_stopping(acc.item(), epoch, max_epoch, phase, train_acc_ci95, loss.item(), model, e_optimizer, 
                           loss_rec, acc_rec, log_dir, log_file_path, i, len(data_loader), 0)
                
            if early_stopping.early_stop:
                print("Early stopping")
                break

        return train_loss_avg, train_acc_avg, e_optimizer, f_optimizer, model

    @staticmethod
    def valid_ssl_multi(epoch, data_loader, model, loss_f, e_optimizer, f_optimizer, max_epoch, log_file_path, log_dir, opt):
        
        criterion_kl = nn.KLDivLoss(size_average=False)
        phase='val'
        test_model = deepcopy(model)  # Copy the model for testing
        test_model.train()  # Set model in training mode to fine-tune in validation loop

        val_accuracies = []
        val_losses = []
        loss_rec = {"train": [], "valid": []}
        acc_rec = {"train": [], "valid": []}
        early_stopping = EarlyStopping(patience=20, verbose=True)

        for i, data in enumerate(tqdm(data_loader(epoch))):

            e_optimizer.zero_grad()
            f_optimizer.zero_grad()
            data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in data]
            data_support = data_support.reshape([-1] + list(data_support.shape[-3:]))
            data_query = data_query.reshape([-1] + list(data_query.shape[-3:]))
            labels_support = labels_support.reshape(-1)
            labels_query = labels_query.reshape(-1)

            for j in range(10):
                outputs_s = test_model(data_support)    
                loss_s = loss_f(outputs_s, labels_support) 
                loss_s.backward()
                # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping 
                e_optimizer.step()
                f_optimizer.step()

            test_model.eval()
            with torch.no_grad():
                outputs_q = test_model(data_query)
                loss_q = loss_f(outputs_q, labels_query)

            # 统计
            acc = count_accuracy(outputs_q, labels_query)
            val_accuracies.append(acc.item())
            val_losses.append(loss_q.item())
            val_acc_avg = np.mean(np.array(val_accuracies))
            val_acc_ci95 = 1.96 * np.std(np.array(val_accuracies)) / np.sqrt(opt.val_episode)
            val_loss_avg = np.mean(np.array(val_losses))    

            if not opt.earlystop_val:
                log(log_file_path, 'Validation Epoch: [{:0>3}/{:0>3}] \t Batch: [{:0>3}/{:0>3}] \t Loss: {:.4f} \t Accuracy: {:.2f} % ({:.2f} %)'
                    .format(epoch, max_epoch, i, len(data_loader), loss_q.item(), val_acc_avg, val_acc_ci95))
            else:
                early_stopping(acc.item(), epoch, max_epoch, phase, val_acc_ci95, loss_q.item(), model, e_optimizer, 
                           loss_rec, acc_rec, log_dir, log_file_path, i, len(data_loader), 0)
                
            if early_stopping.early_stop:
                print("Early stopping")
                break

        return val_loss_avg, val_acc_avg
    
    @staticmethod
    def test_ssl_multi(data_loader, model, loss_f, optimizer, log_file_path, log_dir, attack_num, opt):
        phase='test'
        test_accuracies = []
        test_losses = []
        loss_rec = {"train": [], "valid": []}
        acc_rec = {"train": [], "valid": []}
        best_acc, best_episode = 0.0, 0
        epoch, max_epoch = 0, 0
        early_stopping = EarlyStopping(patience=50, verbose=True)
        timer = time.time()

        for i, data in enumerate(tqdm(data_loader(epoch=0))):
            # data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in data]  # all_test
            data_support, labels_support, data_query, labels_query = [x.cuda() for x in data]          # single_test
            data_support = data_support.reshape([-1] + list(data_support.shape[-3:]))
            data_query = data_query.reshape([-1] + list(data_query.shape[-3:]))
            labels_support = labels_support.reshape(-1)
            labels_query = labels_query.reshape(-1)
            test_model = deepcopy(model)
            for j in range(10):
                outputs_s = test_model(data_support)    
                loss_s = loss_f(outputs_s, labels_support)
                optimizer.zero_grad()   
                # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping 
                loss_s.backward()
                optimizer.step()
                j += 1
                if j >= 10:
                    break
            test_model.eval()
            with torch.no_grad():
                outputs_q = test_model(data_query)
                loss_q = loss_f(outputs_q, labels_query)
            loss = loss_q
            # 统计
            acc = count_accuracy(outputs_q, labels_query)
            test_accuracies.append(acc.item())
            test_losses.append(loss.item())
            # test_acc_avg = np.mean(np.array(test_accuracies))
            test_acc_ci95 = 1.96 * np.std(np.array(test_accuracies)) / np.sqrt(opt.test_episode)
            early_stopping(acc.item(), epoch, max_epoch, phase, test_acc_ci95, loss.item(), model, optimizer, 
                        loss_rec, acc_rec, log_dir, log_file_path, i, len(data_loader), attack_num)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        return acc.item(), loss.item()


    @staticmethod
    # def train_ssl_multi(epoch, data_loader, model, loss_f, e_optimizer, f_optimizer, max_epoch, log_file_path, log_dir, opt):
    def train_ssl_multi_opt_new(epoch, data_loader, model, loss_f, e_optimizer, f_optimizer, lr_schedule_e, lr_schedule_f, max_epoch, log_file_path, log_dir, opt):    
        
        criterion_kl = nn.KLDivLoss(size_average=False)
        phase='train'
        model.train()

        train_accuracies = []
        train_losses = []
        loss_rec = {"train": [], "valid": []}
        acc_rec = {"train": [], "valid": []}
        early_stopping = EarlyStopping(patience=20, verbose=True)

        for i, data in enumerate(tqdm(data_loader(epoch))):

            e_optimizer.zero_grad()
            f_optimizer.zero_grad()

            data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in data]
            data_support = data_support.reshape([-1] + list(data_support.shape[-3:]))
            data_query = data_query.reshape([-1] + list(data_query.shape[-3:]))
            labels_support = labels_support.reshape(-1)
            labels_query = labels_query.reshape(-1)
            
            if opt.mixup:
                data_support_X, labels_support_a, labels_support_b, lam = mixup_data(data_support, labels_support, opt.mixup_alpha)
                data_support_X, labels_support_a, labels_support_b = map(Variable, (data_support_X, labels_support_a, labels_support_b))
                data_query_X, labels_query_a, labels_query_b, lam = mixup_data(data_query, labels_query, opt.mixup_alpha)
                data_query_X, labels_query_a, labels_query_b = map(Variable, (data_query_X, labels_query_a, labels_query_b))
                outputs_s = model(data_support_X)
                loss_natural_s = mixup_criterion(loss_f, outputs_s, labels_support_a, labels_support_b, lam)
                if opt.victim not in ["ACL"]:
                    soft_lables_support = F.one_hot(labels_support, num_classes=outputs_s.size(1)).float()
                    # outputs_s = model(data_support)    
                    loss_robust = (1.0 / data_support.size(0)) * criterion_kl(F.log_softmax(outputs_s, dim=1),
                                                        F.softmax(soft_lables_support, dim=1))
                    loss_s = loss_natural_s + loss_robust
                else:
                    loss_s = loss_natural_s
                with torch.no_grad():
                    outputs_q = model(data_query_X)
                    loss_natural_q = mixup_criterion(loss_f, outputs_q, labels_query_a, labels_query_b, lam)
                    if opt.victim not in ["ACL"]:
                        soft_lables_query = F.one_hot(labels_query, num_classes=outputs_q.size(1)).float()
                        loss_robust = (1.0 / data_query.size(0)) * criterion_kl(F.log_softmax(outputs_q, dim=1),
                                                        F.softmax(soft_lables_query, dim=1))
                        # optimizer.zero_grad()
                        loss_q = loss_natural_q + loss_robust
                    else:
                        loss_q = loss_natural_q
            else:
                outputs_s = model(data_support)
                loss_natural_s = loss_f(outputs_s, labels_support)

                if opt.victim not in ["ACL"]:
                    soft_lables_support = F.one_hot(labels_support, num_classes=outputs_s.size(1)).float()
                    # outputs_s = model(data_support)    
                    loss_robust = (1.0 / data_support.size(0)) * criterion_kl(F.log_softmax(outputs_s, dim=1),
                                                        F.softmax(soft_lables_support, dim=1))
                    loss_s = loss_natural_s + loss_robust
                else:
                    loss_s = loss_natural_s

                with torch.no_grad():
                    outputs_q = model(data_query)
                    loss_natural_q = loss_f(outputs_q, labels_query)

                    if opt.victim not in ["ACL"]:
                        soft_lables_query = F.one_hot(labels_query, num_classes=outputs_q.size(1)).float()
                        loss_robust = (1.0 / data_query.size(0)) * criterion_kl(F.log_softmax(outputs_q, dim=1),
                                                        F.softmax(soft_lables_query, dim=1))
                        loss_q = loss_natural_q + loss_robust
                    else:
                        loss_q = loss_natural_q

            # loss = loss_q
            loss = loss_s * opt.lamda + loss_q * (1 - opt.lamda)
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            e_optimizer.step()
            f_optimizer.step()
            # 统计
            acc = count_accuracy(outputs_q, labels_query)
            train_losses.append(loss.item())
            train_accuracies.append(acc.item())
            train_acc_ci95 = 0
            if (i % 1 == 0):
                train_loss_avg = np.mean(train_losses)
                train_acc_avg = np.mean(np.array(train_accuracies))

            if not opt.earlystop_train:
                log(log_file_path, 'Train Epoch: [{:0>3}/{:0>3}] \t Batch: [{:0>3}/{:0>3}] \t Loss: {:.4f} \t Accuracy: {:.2f} % ({:.2f} %)'
                    .format(epoch, max_epoch, i, len(data_loader), loss.item(), train_acc_avg, acc))
            else:
                early_stopping(acc.item(), epoch, max_epoch, phase, train_acc_ci95, loss.item(), model, e_optimizer, 
                           loss_rec, acc_rec, log_dir, log_file_path, i, len(data_loader), 0)
                
            if early_stopping.early_stop:
                print("Early stopping")
                break

        return train_loss_avg, train_acc_avg, e_optimizer, f_optimizer, model
    
    @staticmethod
    # def train_ssl_multi(epoch, data_loader, model, loss_f, e_optimizer, f_optimizer, max_epoch, log_file_path, log_dir, opt):
    def train_ssl_multi_cross(epoch, data_loader, model, loss_f, e_optimizer, f_optimizer, lr_schedule_e, lr_schedule_f, max_epoch, log_file_path, log_dir, opt):    
        
        criterion_kl = nn.KLDivLoss(size_average=False)
        phase='train'
        model.train()

        train_accuracies = []
        train_losses = []
        loss_rec = {"train": [], "valid": []}
        acc_rec = {"train": [], "valid": []}
        early_stopping = EarlyStopping(patience=20, verbose=True)

        embedding_s = []
        label_s = []
        embedding_q = []
        label_q = []
        for i, data in enumerate(tqdm(data_loader(epoch))):

            epoch_now = (epoch-1) + (i + 1) / len(data_loader)
            lr_e = lr_schedule_e(epoch_now)
            e_optimizer.param_groups[0].update(lr=lr_e)
            lr_f = lr_schedule_f(epoch_now)
            f_optimizer.param_groups[0].update(lr=lr_f)
            e_optimizer.zero_grad()
            f_optimizer.zero_grad()

            data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in data]
            data_support = data_support.reshape([-1] + list(data_support.shape[-3:]))
            data_query = data_query.reshape([-1] + list(data_query.shape[-3:]))
            labels_support = labels_support.reshape(-1)
            labels_query = labels_query.reshape(-1)
            
            if opt.mixup:
                data_support_X, labels_support_a, labels_support_b, lam = mixup_data(data_support, labels_support, opt.mixup_alpha)
                data_support_X, labels_support_a, labels_support_b = map(Variable, (data_support_X, labels_support_a, labels_support_b))
                data_query_X, labels_query_a, labels_query_b, lam = mixup_data(data_query, labels_query, opt.mixup_alpha)
                data_query_X, labels_query_a, labels_query_b = map(Variable, (data_query_X, labels_query_a, labels_query_b))
                outputs_s = model(data_support_X)
                loss_natural_s = mixup_criterion(loss_f, outputs_s, labels_support_a, labels_support_b, lam)
                if opt.victim not in ["ACL"]:
                    soft_lables_support = F.one_hot(labels_support, num_classes=outputs_s.size(1)).float()
                    # outputs_s = model(data_support)    
                    loss_robust = (1.0 / data_support.size(0)) * criterion_kl(F.log_softmax(outputs_s, dim=1),
                                                        F.softmax(soft_lables_support, dim=1))
                    loss_s = loss_natural_s + loss_robust
                else:
                    loss_s = loss_natural_s
                with torch.no_grad():
                    outputs_q = model(data_query_X)
                    loss_natural_q = mixup_criterion(loss_f, outputs_q, labels_query_a, labels_query_b, lam)
                    if opt.victim not in ["ACL"]:
                        soft_lables_query = F.one_hot(labels_query, num_classes=outputs_q.size(1)).float()
                        loss_robust = (1.0 / data_query.size(0)) * criterion_kl(F.log_softmax(outputs_q, dim=1),
                                                        F.softmax(soft_lables_query, dim=1))
                        # optimizer.zero_grad()
                        loss_q = loss_natural_q + loss_robust
                    else:
                        loss_q = loss_natural_q
            else:
                outputs_s = model(data_support)
                emb = model(data_support, return_features=True).detach().cpu().numpy()
                embedding_s.append(emb)
                label_s.append(labels_support.cpu().numpy())
                loss_natural_s = loss_f(outputs_s, labels_support)
                # Concatenate all embeddings and labels
                embedding = np.concatenate(embedding_s, axis=0)
                label = np.concatenate(label_s, axis=0)
                if opt.victim not in ["ACL"]:
                    soft_lables_support = F.one_hot(labels_support, num_classes=outputs_s.size(1)).float()
                    # outputs_s = model(data_support)    
                    loss_robust = (1.0 / data_support.size(0)) * criterion_kl(F.log_softmax(outputs_s, dim=1),
                                                        F.softmax(soft_lables_support, dim=1))
                    loss_s = loss_natural_s + loss_robust
                else:
                    loss_s = loss_natural_s

                with torch.no_grad():
                    outputs_q = model(data_query)
                    loss_natural_q = loss_f(outputs_q, labels_query)
                    emb = model(data_query, return_features=True).detach().cpu().numpy()
                    embedding_q.append(emb)
                    label_q.append(labels_query.cpu().numpy())
                    if opt.victim not in ["ACL"]:
                        soft_lables_query = F.one_hot(labels_query, num_classes=outputs_q.size(1)).float()
                        loss_robust = (1.0 / data_query.size(0)) * criterion_kl(F.log_softmax(outputs_q, dim=1),
                                                        F.softmax(soft_lables_query, dim=1))
                        loss_q = loss_natural_q + loss_robust
                    else:
                        loss_q = loss_natural_q
                # Concatenate all embeddings and labels
                embedding = np.concatenate(embedding_q, axis=0)
                label = np.concatenate(label_q, axis=0)    
            # Compute Intra-Class Variance

            intra_variance = compute_intra_class_variance(embedding, label, outputs_s.size(1))
            print(f"Intra-Class Variance: {intra_variance}")
            # Compute Inter-Class Variance
            inter_variance = compute_inter_class_variance(embedding, label, outputs_s.size(1))
            print(f"Inter-Class Variance: {inter_variance}")
            # loss = loss_q
            loss = loss_s * opt.lamda + loss_q * (1 - opt.lamda) + 1 * intra_variance - 1 * inter_variance
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            e_optimizer.step()
            f_optimizer.step()
            # 统计
            acc = count_accuracy(outputs_q, labels_query)
            train_losses.append(loss.item())
            train_accuracies.append(acc.item())
            train_acc_ci95 = 0
            if (i % 1 == 0):
                train_loss_avg = np.mean(train_losses)
                train_acc_avg = np.mean(np.array(train_accuracies))

            if not opt.earlystop_train:
                log(log_file_path, 'Train Epoch: [{:0>3}/{:0>3}] \t Batch: [{:0>3}/{:0>3}] \t Loss: {:.4f} \t Accuracy: {:.2f} % ({:.2f} %)'
                    .format(epoch, max_epoch, i, len(data_loader), loss.item(), train_acc_avg, acc))
            else:
                early_stopping(acc.item(), epoch, max_epoch, phase, train_acc_ci95, loss.item(), model, e_optimizer, 
                            loss_rec, acc_rec, log_dir, log_file_path, i, len(data_loader), 0)
                
            if early_stopping.early_stop:
                print("Early stopping")
                break

        return train_loss_avg, train_acc_avg, e_optimizer, f_optimizer, model

    @staticmethod
    def valid_ssl_multi_cross(epoch, data_loader, model, loss_f, e_optimizer, f_optimizer, max_epoch, log_file_path, log_dir, opt):
        
        criterion_kl = nn.KLDivLoss(size_average=False)
        phase='val'
        test_model = deepcopy(model)  # Copy the model for testing
        test_model.train()  # Set model in training mode to fine-tune in validation loop

        val_accuracies = []
        val_losses = []
        # embedding_s = []
        # label_s = []
        loss_rec = {"train": [], "valid": []}
        acc_rec = {"train": [], "valid": []}
        early_stopping = EarlyStopping(patience=20, verbose=True)

        for i, data in enumerate(tqdm(data_loader(epoch))):

            e_optimizer.zero_grad()
            f_optimizer.zero_grad()
            data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in data]
            data_support = data_support.reshape([-1] + list(data_support.shape[-3:]))
            data_query = data_query.reshape([-1] + list(data_query.shape[-3:]))
            labels_support = labels_support.reshape(-1)
            labels_query = labels_query.reshape(-1)

            for j in range(10):
                outputs_s = test_model(data_support)

                # emb = test_model(data_support, return_features=True).detach().cpu().numpy()
                # embedding_s.append(emb)    
                # label_s.append(labels_support.cpu().numpy())
                # # Concatenate all embeddings and labels
                # embedding = np.concatenate(embedding_s, axis=0)
                # label = np.concatenate(label_s, axis=0)
                # intra_variance = compute_intra_class_variance(embedding, label, outputs_s.size(1))
                # print(f"Intra-Class Variance: {intra_variance}")
                # # Compute Inter-Class Variance
                # inter_variance = compute_inter_class_variance(embedding, label, outputs_s.size(1))
                # print(f"Inter-Class Variance: {inter_variance}")

                loss_s = loss_f(outputs_s, labels_support) 
                # loss_s = loss_s + 0.1 * intra_variance - 0.1 * inter_variance
                loss_s.backward()
                # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping 
                e_optimizer.step()
                f_optimizer.step()
            
            test_model.eval()
            with torch.no_grad():
                outputs_q = test_model(data_query)
                loss_q = loss_f(outputs_q, labels_query)

            # 统计
            acc = count_accuracy(outputs_q, labels_query)
            val_accuracies.append(acc.item())
            val_losses.append(loss_q.item())
            val_acc_avg = np.mean(np.array(val_accuracies))
            val_acc_ci95 = 1.96 * np.std(np.array(val_accuracies)) / np.sqrt(opt.val_episode)
            val_loss_avg = np.mean(np.array(val_losses))    

            if not opt.earlystop_val:
                log(log_file_path, 'Validation Epoch: [{:0>3}/{:0>3}] \t Batch: [{:0>3}/{:0>3}] \t Loss: {:.4f} \t Accuracy: {:.2f} % ({:.2f} %)'
                    .format(epoch, max_epoch, i, len(data_loader), loss_q.item(), val_acc_avg, val_acc_ci95))
            else:
                early_stopping(acc.item(), epoch, max_epoch, phase, val_acc_ci95, loss_q.item(), model, e_optimizer, 
                            loss_rec, acc_rec, log_dir, log_file_path, i, len(data_loader), 0)
                
            if early_stopping.early_stop:
                print("Early stopping")
                break

        return val_loss_avg, val_acc_avg

    @staticmethod
    def test_ssl_multi_cross(data_loader, model, loss_f, optimizer, log_file_path, log_dir, attack_num, opt):
        phase='test'
        test_accuracies = []
        test_losses = []
        loss_rec = {"train": [], "valid": []}
        acc_rec = {"train": [], "valid": []}
        best_acc, best_episode = 0.0, 0
        epoch, max_epoch = 0, 0
        early_stopping = EarlyStopping(patience=50, verbose=True)
        timer = time.time()

        for i, data in enumerate(tqdm(data_loader(epoch=0))):
            # data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in data]  # all_test
            data_support, labels_support, data_query, labels_query = [x.cuda() for x in data]          # single_test
            data_support = data_support.reshape([-1] + list(data_support.shape[-3:]))
            data_query = data_query.reshape([-1] + list(data_query.shape[-3:]))
            labels_support = labels_support.reshape(-1)
            labels_query = labels_query.reshape(-1)
            test_model = deepcopy(model)
            for j in range(10):
                outputs_s = test_model(data_support)    
                loss_s = loss_f(outputs_s, labels_support)
                optimizer.zero_grad()   
                # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping 
                loss_s.backward()
                optimizer.step()
                j += 1
                if j >= 10:
                    break
            test_model.eval()
            with torch.no_grad():
                outputs_q = test_model(data_query)
                loss_q = loss_f(outputs_q, labels_query)
            loss = loss_q
            # 统计
            acc = count_accuracy(outputs_q, labels_query)
            test_accuracies.append(acc.item())
            test_losses.append(loss.item())
            # test_acc_avg = np.mean(np.array(test_accuracies))
            test_acc_ci95 = 1.96 * np.std(np.array(test_accuracies)) / np.sqrt(opt.test_episode)
            early_stopping(acc.item(), epoch, max_epoch, phase, test_acc_ci95, loss.item(), model, optimizer, 
                        loss_rec, acc_rec, log_dir, log_file_path, i, len(data_loader), attack_num)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        return acc.item(), loss.item()

    @staticmethod
    def attack(input_adv, input_per, label_num, label, atk_key, atk):
        if  eval(atk_key) <= 12: 
            adv_examples = atk.generate(x=input_adv.numpy(), y=label_num)
        elif 13 <= eval(atk_key) <= 29: 
            adv_examples = atk(input_adv, label)
        # elif 30 <= eval(atk_key) <= 47: 
        #     params = NOISE_PARAMS[atk][random.randint(0,4)]
        #     adv_examples = PERTURBATIONS[atk](input_per, severity_params=params, image_size=input_per.size[0])
        #     if not isinstance(adv_examples, np.ndarray):
        #         adv_examples = np.asarray(adv_examples)
        #     adv_examples = np.expand_dims(adv_examples.transpose(2, 0, 1),axis=0)   
        return adv_examples

    @staticmethod
    # def train_ssl_multi(epoch, data_loader, model, loss_f, e_optimizer, f_optimizer, max_epoch, log_file_path, log_dir, opt):
    def train_ssl_encoder(epoch, data_loader, model, encoder, criterion_C, loss_f, e_optimizer, f_optimizer, lr_schedule_e, lr_schedule_f, max_epoch, log_file_path, log_dir, opt):    
        
        criterion_kl = nn.KLDivLoss(size_average=False)
        phase='train'
        model.train()

        train_accuracies = []
        train_losses = []
        loss_rec = {"train": [], "valid": []}
        acc_rec = {"train": [], "valid": []}
        early_stopping = EarlyStopping(patience=20, verbose=True)

        for i, data in enumerate(tqdm(data_loader(epoch))):

            epoch_now = (epoch-1) + (i + 1) / len(data_loader)
            lr_e = lr_schedule_e(epoch_now)
            e_optimizer.param_groups[0].update(lr=lr_e)
            lr_f = lr_schedule_f(epoch_now)
            f_optimizer.param_groups[0].update(lr=lr_f)
            e_optimizer.zero_grad()
            f_optimizer.zero_grad()

            data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in data]
            data_support = data_support.reshape([-1] + list(data_support.shape[-3:]))
            data_query = data_query.reshape([-1] + list(data_query.shape[-3:]))
            labels_support = labels_support.reshape(-1)
            labels_query = labels_query.reshape(-1)
            
            if opt.mixup:
                data_support_X, labels_support_a, labels_support_b, lam = mixup_data(data_support, labels_support, opt.mixup_alpha)
                data_support_X, labels_support_a, labels_support_b = map(Variable, (data_support_X, labels_support_a, labels_support_b))
                data_query_X, labels_query_a, labels_query_b, lam = mixup_data(data_query, labels_query, opt.mixup_alpha)
                data_query_X, labels_query_a, labels_query_b = map(Variable, (data_query_X, labels_query_a, labels_query_b))
                outputs_s = model(data_support_X)
                loss_natural_s = mixup_criterion(loss_f, outputs_s, labels_support_a, labels_support_b, lam)
                if opt.victim not in ["ACL"]:
                    soft_lables_support = F.one_hot(labels_support, num_classes=outputs_s.size(1)).float()
                    # outputs_s = model(data_support)    
                    loss_robust = (1.0 / data_support.size(0)) * criterion_kl(F.log_softmax(outputs_s, dim=1),
                                                        F.softmax(soft_lables_support, dim=1))
                    loss_s = loss_natural_s + loss_robust
                else:
                    loss_s = loss_natural_s
                with torch.no_grad():
                    outputs_q = model(data_query_X)
                    loss_natural_q = mixup_criterion(loss_f, outputs_q, labels_query_a, labels_query_b, lam)
                    if opt.victim not in ["ACL"]:
                        soft_lables_query = F.one_hot(labels_query, num_classes=outputs_q.size(1)).float()
                        loss_robust = (1.0 / data_query.size(0)) * criterion_kl(F.log_softmax(outputs_q, dim=1),
                                                        F.softmax(soft_lables_query, dim=1))
                        # optimizer.zero_grad()
                        loss_q = loss_natural_q + loss_robust
                    else:
                        loss_q = loss_natural_q
            else:
                # features_s = encoder(data_support)
                outputs_s = model(data_support)
                # loss_natural_s = loss_f(features_s, t=0.2)
                loss_natural_s = criterion_C(outputs_s, labels_support)

                if opt.victim not in ["ACL"]:
                    soft_lables_support = F.one_hot(labels_support, num_classes=outputs_s.size(1)).float()
                    # outputs_s = model(data_support)    
                    loss_robust = (1.0 / data_support.size(0)) * criterion_kl(F.log_softmax(outputs_s, dim=1),
                                                        F.softmax(soft_lables_support, dim=1))
                    loss_s = loss_natural_s + loss_robust
                else:
                    loss_s = loss_natural_s

                with torch.no_grad():
                    # features_q = encoder(data_query)
                    outputs_q = model(data_query)
                    # loss_natural_q = loss_f(features_q, t=0.2)
                    loss_natural_q = criterion_C(outputs_q, labels_query)

                    if opt.victim not in ["ACL"]:
                        soft_lables_query = F.one_hot(labels_query, num_classes=outputs_q.size(1)).float()
                        loss_robust = (1.0 / data_query.size(0)) * criterion_kl(F.log_softmax(outputs_q, dim=1),
                                                        F.softmax(soft_lables_query, dim=1))
                        loss_q = loss_natural_q + loss_robust
                    else:
                        loss_q = loss_natural_q
            # # Assuming 'features_s' and 'features_q' are the feature representations from the support and query sets
            # features_combined = torch.cat((features_s, features_q), dim=0)  # Concatenate the support and query features
            # # Compute the contrastive loss using nt_xent
            # loss_sq = loss_f(features_combined, t=0.5)  # Use NT-Xent to compute the contrastive loss
            # loss = loss_s * opt.lamda + loss_q * (1 - opt.lamda) + loss_sq/21
            loss = loss_s * opt.lamda + loss_q * (1 - opt.lamda)
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            e_optimizer.step()
            f_optimizer.step()
            # 统计
            acc = count_accuracy(outputs_q, labels_query)
            train_losses.append(loss.item())
            train_accuracies.append(acc.item())
            train_acc_ci95 = 0
            if (i % 1 == 0):
                train_loss_avg = np.mean(train_losses)
                train_acc_avg = np.mean(np.array(train_accuracies))

            if not opt.earlystop_train:
                log(log_file_path, 'Train Epoch: [{:0>3}/{:0>3}] \t Batch: [{:0>3}/{:0>3}] \t Loss: {:.4f} \t Accuracy: {:.2f} % ({:.2f} %)'
                    .format(epoch, max_epoch, i, len(data_loader), loss.item(), train_acc_avg, acc))
            else:
                early_stopping(acc.item(), epoch, max_epoch, phase, train_acc_ci95, loss.item(), model, e_optimizer, 
                           loss_rec, acc_rec, log_dir, log_file_path, i, len(data_loader), 0)
                
            if early_stopping.early_stop:
                print("Early stopping")
                break

        return train_loss_avg, train_acc_avg, e_optimizer, f_optimizer, model, encoder
    
    @staticmethod
    def train_ssl_clean_SAR(epoch, data_loader, model, loss_f, optimizer, lr_schedule, max_epoch, log_file_path, log_dir, opt):
        
        criterion_kl = nn.KLDivLoss(size_average=False)
        phase='train'
        model.train()

        train_accuracies = []
        train_losses = []
        loss_rec = {"train": [], "valid": []}
        acc_rec = {"train": [], "valid": []}
        early_stopping = EarlyStopping(patience=20, verbose=True)

        for i, data in enumerate(tqdm(data_loader(epoch))):

            epoch_now = (epoch-1) + (i + 1) / len(data_loader)
            lr = lr_schedule(epoch_now)
            optimizer.param_groups[0].update(lr=lr)

            data_support, labels_support, clean_support, data_query, labels_query, clean_query, _, _ = [x.cuda() for x in data]
            data_support = data_support.reshape([-1] + list(data_support.shape[-3:]))
            data_query = data_query.reshape([-1] + list(data_query.shape[-3:]))
            clean_support = clean_support.reshape([-1] + list(clean_support.shape[-3:]))
            clean_query = clean_query.reshape([-1] + list(clean_query.shape[-3:]))
            labels_support = labels_support.reshape(-1)
            labels_query = labels_query.reshape(-1)

            optimizer.zero_grad()
            if opt.mixup:

                data_support_X, labels_support_a, labels_support_b, lam = mixup_data(data_support, labels_support, opt.mixup_alpha)
                data_support_X, labels_support_a, labels_support_b = map(Variable, (data_support_X, labels_support_a, labels_support_b))
                data_query_X, labels_query_a, labels_query_b, lam = mixup_data(data_query, labels_query, opt.mixup_alpha)
                data_query_X, labels_query_a, labels_query_b = map(Variable, (data_query_X, labels_query_a, labels_query_b))

                outputs_s = model(data_support_X)
                soft_lables_support = F.one_hot(labels_support, num_classes=outputs_s.size(1)).float()
                # outputs_s = model(data_support)    
                loss_natural_s = mixup_criterion(loss_f, outputs_s, labels_support_a, labels_support_b, lam)
                loss_robust = (1.0 / data_support.size(0)) * criterion_kl(F.log_softmax(outputs_s, dim=1),
                                                    F.softmax(soft_lables_support, dim=1))
                loss_s = loss_natural_s + loss_robust

                with torch.no_grad():
                    outputs_q = model(data_query_X)
                    soft_lables_query = F.one_hot(labels_query, num_classes=outputs_q.size(1)).float()
                    loss_natural_q = mixup_criterion(loss_f, outputs_q, labels_query_a, labels_query_b, lam)
                    loss_robust = (1.0 / data_query.size(0)) * criterion_kl(F.log_softmax(outputs_q, dim=1),
                                                    F.softmax(soft_lables_query, dim=1))
                    # optimizer.zero_grad()
                    loss_q = loss_natural_q + loss_robust

            else:
                outputs_s = model(data_support)
                loss_natural_s = loss_f(outputs_s, labels_support)

                if opt.victim not in ["ACL"]:
                    outputs_s_c = model(clean_support)
                    loss_robust = (1.0 / data_support.size(0)) * criterion_kl(F.log_softmax(outputs_s, dim=1),
                                                        F.softmax(outputs_s_c, dim=1))
                    loss_s = loss_natural_s + loss_robust
                else:
                    loss_s = loss_natural_s

                with torch.no_grad():
                    outputs_q = model(data_query)
                    loss_natural_q = loss_f(outputs_q, labels_query)

                    if opt.victim not in ["ACL"]:
                        outputs_q_c = model(clean_query)
                        loss_robust = (1.0 / data_query.size(0)) * criterion_kl(F.log_softmax(outputs_q, dim=1),
                                                            F.softmax(outputs_q_c, dim=1))
                        loss_s = loss_natural_q + loss_robust
                    else:
                        loss_q = loss_natural_q

            # loss = loss_q
            loss = loss_s * opt.lamda + loss_q * (1 - opt.lamda)

            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            # 统计
            acc = count_accuracy(outputs_q, labels_query)
            train_losses.append(loss.item())
            train_accuracies.append(acc.item())
            train_acc_ci95 = 0

            if (i % 1 == 0):
                train_loss_avg = np.mean(train_losses)
                train_acc_avg = np.mean(np.array(train_accuracies))
            if not opt.earlystop_train:
                log(log_file_path, 'Train Epoch: [{:0>3}/{:0>3}] \t Batch: [{:0>3}/{:0>3}] \t Loss: {:.4f} \t Accuracy: {:.2f} % ({:.2f} %)'
                    .format(epoch, max_epoch, i, len(data_loader), loss.item(), train_acc_avg, acc))
            else:
                early_stopping(acc.item(), epoch, max_epoch, phase, train_acc_ci95, loss.item(), model, optimizer, 
                           loss_rec, acc_rec, log_dir, log_file_path, i, len(data_loader), 0)
                
            if early_stopping.early_stop:
                print("Early stopping")
                break

        return train_loss_avg, train_acc_avg, lr, optimizer, model
        
def show_confMat(confusion_mat, classes, set_name, out_dir, verbose=False):
    """
    混淆矩阵绘制
    :param confusion_mat:
    :param classes: 类别名
    :param set_name: trian/valid
    :param out_dir:
    :return:
    """
    cls_num = len(classes)
    # 归一化
    confusion_mat_N = confusion_mat.copy()
    for i in range(len(classes)):
        confusion_mat_N[i, :] = confusion_mat[i, :] / confusion_mat[i, :].sum()

    # 获取颜色
    cmap = plt.cm.get_cmap('Greys')  # 更多颜色: http://matplotlib.org/examples/color/colormaps_reference.html
    plt.imshow(confusion_mat_N, cmap=cmap)
    plt.colorbar()

    # 设置文字
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, list(classes), rotation=60)
    plt.yticks(xlocations, list(classes))
    plt.xlabel('Predict label')
    plt.ylabel('True label')
    plt.title('Confusion_Matrix_' + set_name)

    # 打印数字
    for i in range(confusion_mat_N.shape[0]):
        for j in range(confusion_mat_N.shape[1]):
            plt.text(x=j, y=i, s=int(confusion_mat[i, j]), va='center', ha='center', color='red', fontsize=10)
    # 显示

    plt.savefig(os.path.join(out_dir, 'Confusion_Matrix' + set_name + '.png'))
    # plt.show()
    plt.close("all")
    # plt.close()

    if verbose:
        for i in range(cls_num):
            print('class:{:<10}, total num:{:<6}, correct num:{:<5}  Recall: {:.2%} Precision: {:.2%}'.format(
                classes[i], np.sum(confusion_mat[i, :]), confusion_mat[i, i],
                confusion_mat[i, i] / (.1 + np.sum(confusion_mat[i, :])),
                confusion_mat[i, i] / (.1 + np.sum(confusion_mat[:, i]))))


# def plot_line(train_x, train_y, valid_x, valid_y, mode, out_dir):
def plot_line(train_x, 
                        A_MAT, 
                        # A_AT, 
                        # R_MAT, 
                        # R_AT, 
                        mode, out_dir):
    """
    绘制训练和验证集的loss曲线/acc曲线
    :param train_x: epoch
    :param train_y: 标量值
    :param valid_x:
    :param valid_y:
    :param mode:  'loss' or 'acc'
    :param out_dir:
    :return:        A_MAT_loss_rec["train"], plt_x, A_MAT_loss_rec["valid"],
                    A_AT_loss_rec["train"], plt_x, A_AT_loss_rec["valid"],
                    R_MAT_loss_rec["train"], plt_x, R_MAT_loss_rec["valid"],
                    R_AT_loss_rec["train"], plt_x, R_AT_loss_rec["valid"],
    """
    # plt.plot(train_x, A_AT, linewidth =5.0, label='AT_MNIST')
    # plt.plot(train_x, A_MAT, linewidth =5.0, label='Meta-AT_MNIST')
    plt.rcParams.update({'font.size': 14})
    # plt.plot(train_x, R_AT, linewidth =5.0, label='AT_CIFAR_10')
    plt.plot(train_x, A_MAT, linewidth =5.0, label='Meta-AT_CIFAR_10')
    
    plt.ylabel(str(mode))
    plt.xlabel('Epoch')

    # location = 'upper right' 
    location = 'best' 
    # if mode == 'loss' else 'upper left'
    plt.legend(loc=location)

    plt.title('_'.join([mode]))
    plt.savefig(os.path.join(out_dir, mode + '.png'))
    plt.close("all")
    # plt.close()

def imshow(img, title):
    npimg = img.numpy()
    fig = plt.figure(figsize=(5, 15))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.axis('off')
    plt.show()

