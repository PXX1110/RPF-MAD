import sys
sys.path.append(r'/home/users/pxx/workplace/5Adversarial/4Meta_AT_ResNet')
import numpy as np
from art.attacks.evasion import AutoProjectedGradientDescent
from torchattacks.attacks import autoattack
from torchattacks.attacks.fab import FAB
from torchattacks.attacks.square import Square
from torchattacks.attacks.apgdt import APGDT

def get_attack_CIFAER10(classifier, model):
    ATKS = {
        # "ART"
        "0": AutoProjectedGradientDescent(estimator=classifier, eps=8/255, batch_size=1000, loss_type = "cross_entropy"), # [Linf/L2-W-2020]
        "1": AutoProjectedGradientDescent(estimator=classifier, eps=8/255, batch_size=1000, loss_type = "difference_logits_ratio"),# [Linf/L2-W-2020]
        "2": AutoProjectedGradientDescent(estimator=classifier, eps=8/255, batch_size=1000, targeted=True, nb_random_init=1, loss_type = "cross_entropy"),# [Linf/L2-W-2020]
        "3": AutoProjectedGradientDescent(estimator=classifier, eps=8/255, batch_size=1000, targeted=True, loss_type = "difference_logits_ratio"),# [Linf/L2-W-2020]
        "4": FAB(model, eps=8/255, steps=100, n_restarts=5, targeted=False),          # [Linf/L2/L1-W-2020]
        "5": FAB(model, eps=8/255, steps=100, n_restarts=5, targeted=True),          # [Linf/L2/L1-W-2020]
        "6": Square(model, eps=8/255, n_queries=5000, n_restarts=1, loss='ce'),     # [Linf/L2-B-2020]
        "7": APGDT(model, eps=8/255, n_classes=10, n_restarts=1),     # [Linf/L2-B-2020]
            }
    return ATKS

def get_attack_MNIST(classifier, model):
    ATKS = {
     
            }
    return ATKS


def get_attack_TinyImageNet(classifier, model):
    ATKS = {
        
            }
    return ATKS
