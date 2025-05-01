import sys
sys.path.append(r'/home/users/pxx/workplace/5Adversarial/4Meta_AT_ResNet')
import numpy as np
from art.attacks.evasion import BoundaryAttack,ElasticNet,GeoDA,HopSkipJump,NewtonFool,SaliencyMapMethod,ShadowAttack
from art.attacks.evasion import SimBA,SpatialTransformation,UniversalPerturbation,Wasserstein,ZooAttack,DeepFool
from torchattacks.attacks.fgsm import FGSM
from torchattacks.attacks.ffgsm import FFGSM
from torchattacks.attacks.rfgsm import RFGSM
from torchattacks.attacks.mifgsm import MIFGSM
from torchattacks.attacks.tifgsm import TIFGSM
from torchattacks.attacks.bim import BIM
from torchattacks.attacks.cw import CW
from torchattacks.attacks.pgd import PGD
from torchattacks.attacks.pgdl2 import PGDL2
from torchattacks.attacks.eotpgd import EOTPGD
from torchattacks.attacks.tpgd import TPGD
from torchattacks.attacks.apgd import APGD
from torchattacks.attacks.fab import FAB
from torchattacks.attacks.square import Square
from torchattacks.attacks.onepixel import OnePixel


def get_attack_CIFAER10(classifier, model):
    ATKS = {
        # "ART"
        "0": SaliencyMapMethod(classifier=classifier, verbose=False),       # [L0-W-2015]
        "1": DeepFool(classifier=classifier, max_iter=100, verbose=False),                # [L2-W-2016]
        "2": UniversalPerturbation(classifier=classifier, verbose=False),   # [Linf-W-2017]
        "3": NewtonFool(classifier=classifier, verbose=False),              # [L0/L2-W-2017]
        "4": BoundaryAttack(estimator=classifier, max_iter=100, verbose=False),           # [L2-B-2018]
        "5": ElasticNet(classifier=classifier, verbose=False),              # [L1-W-2018]
        "6": ZooAttack(classifier=classifier, max_iter=30, verbose=False),               # [L0-B-2018]
        "7": SpatialTransformation(classifier=classifier, max_rotation=60.0, verbose=False),   # [L?-B-2019]
        "8": HopSkipJump(classifier=classifier, verbose=False),             # [L2/Linf-B-2020]
        "9": SimBA(classifier=classifier),                   # [L2-B-2020]
        "10": ShadowAttack(estimator=classifier, lambda_tv=0.006, verbose=False),            # [L?-W-2020]
        "11": GeoDA(estimator=classifier, verbose=False),                                  # [Linf-B-2020]
        "12": Wasserstein(estimator=classifier, regularization=20.0, max_iter=100, verbose=False),                            # [L?-W-2020]
        # "TA"
        "13": FGSM(model, eps=8/255),                                                # [Linf-W-2014]
        "14": BIM(model, eps=8/255, alpha=2/255, steps=50),                          # [Linf-W-2017] steps=100
        "15": CW(model, c=1, lr=0.01, steps=50, kappa=0),                            # [Linf-W-2017]
        "16": MIFGSM(model, eps=8/255, alpha=2/255, steps=50, decay=0.1),            # [Linf-W-2018]
        "17": TIFGSM(model, eps=8/255, alpha=2/255, steps=50, decay=0.1),            # [Linf-W-2019]
        "18": PGD(model, eps=8/255, alpha=2/225, steps=50, random_start=True),       # [Linf-W-2019]
        "19": PGDL2(model, eps=1, alpha=0.2, steps=50),                              # [L2-W-2019]
        "20": TPGD(model, eps=8/255, alpha=2/255, steps=50),                         # [Linf-W-2019]
        "21": RFGSM(model, eps=8/255, alpha=2/255, steps=50),                        # [Linf-W-2020]
        "22": APGD(model, eps=8/255, steps=50, eot_iter=1, n_restarts=1, loss='ce'), # [Linf/L2-W-2020]
        "23": APGD(model, eps=8/255, steps=50, eot_iter=1, n_restarts=1, loss='dlr'),# [Linf/L2-W-2020]
        "24": FFGSM(model, eps=8/255, alpha=10/255),                                 # [Linf-W-2020]
        "25": Square(model, eps=8/255, n_queries=2500, n_restarts=1, loss='ce'),     # [Linf/L2-B-2020]
        "26": TIFGSM(model, eps=8/255, alpha=2/255, steps=50, diversity_prob=0.5, IOK=0.9), # [Linf-W-2019] steps=100
        "27": EOTPGD(model, eps=8/255, alpha=2/255, steps=50, eot_iter=2),                          # [Linf-W-2019]
        "28": OnePixel(model),                                                                      # [L0-B-2019]
        "29": FAB(model, eps=8/255, steps=100, n_restarts=1, targeted=False),          # [Linf/L2/L1-W-2020]
        # "NOISE" 
        "30": "Gaussian Noise",
        "31": "Shot Noise",
        "32": "Impulse Noise",
        "33": "Defocus Blur",
        "34": "Glass Blur",
        "35": "Motion Blur",
        "36": "Zoom Blur",
        "37": "Snow",
        "38": "Frost",
        "39": "Fog",
        "40": "Brightness",
        "41": "Contrast",
        "42": "Elastic",
        "43": "Pixelate",
        "44": "JPEG",
        "45": "Speckle Noise",
        "46": "Gaussian Blur",
        "47": "Saturate"
            }
    return ATKS

def get_attack_GTSRB(classifier, model):
    ATKS = {
        # "ART"
        "0": SaliencyMapMethod(classifier=classifier, verbose=False),       # [L0-W-2015]
        "1": DeepFool(classifier=classifier, max_iter=100, verbose=False),                # [L2-W-2016]
        "2": UniversalPerturbation(classifier=classifier, verbose=False),   # [Linf-W-2017]
        "3": NewtonFool(classifier=classifier, verbose=False),              # [L0/L2-W-2017]
        "4": BoundaryAttack(estimator=classifier, max_iter=100, verbose=False),           # [L2-B-2018]
        "5": ElasticNet(classifier=classifier, verbose=False),              # [L1-W-2018]
        "6": ZooAttack(classifier=classifier, max_iter=30, verbose=False),               # [L0-B-2018]
        "7": SpatialTransformation(classifier=classifier, max_rotation=60.0, verbose=False),   # [L?-B-2019]
        "8": HopSkipJump(classifier=classifier, verbose=False),             # [L2/Linf-B-2020]
        "9": SimBA(classifier=classifier),                   # [L2-B-2020]
        "10": ShadowAttack(estimator=classifier, lambda_tv=0.006, verbose=False),            # [L?-W-2020]
        "11": GeoDA(estimator=classifier, verbose=False),                                  # [Linf-B-2020]
        "12": Wasserstein(estimator=classifier, regularization=20.0, max_iter=100, verbose=False),                            # [L?-W-2020]
        # "TA"
        "13": FGSM(model, eps=8/255),                                                # [Linf-W-2014]
        "14": BIM(model, eps=8/255, alpha=2/255, steps=50),                          # [Linf-W-2017] steps=100
        "15": CW(model, c=1, lr=0.01, steps=50, kappa=0),                            # [Linf-W-2017]
        "16": MIFGSM(model, eps=8/255, alpha=2/255, steps=50, decay=0.1),            # [Linf-W-2018]
        "17": TIFGSM(model, eps=8/255, alpha=2/255, steps=50, decay=0.1),            # [Linf-W-2019]
        "18": PGD(model, eps=8/255, alpha=2/225, steps=50, random_start=True),       # [Linf-W-2019]
        "19": PGDL2(model, eps=1, alpha=0.2, steps=50),                              # [L2-W-2019]
        "20": TPGD(model, eps=8/255, alpha=2/255, steps=50),                         # [Linf-W-2019]
        "21": RFGSM(model, eps=8/255, alpha=2/255, steps=50),                        # [Linf-W-2020]
        "22": APGD(model, eps=8/255, steps=50, eot_iter=1, n_restarts=1, loss='ce'), # [Linf/L2-W-2020]
        "23": APGD(model, eps=8/255, steps=50, eot_iter=1, n_restarts=1, loss='dlr'),# [Linf/L2-W-2020]
        "24": FFGSM(model, eps=8/255, alpha=10/255),                                 # [Linf-W-2020]
        "25": Square(model, eps=8/255, n_queries=2500, n_restarts=1, loss='ce'),     # [Linf/L2-B-2020]
        "26": TIFGSM(model, eps=8/255, alpha=2/255, steps=50, diversity_prob=0.5), # [Linf-W-2019] steps=100
        "27": EOTPGD(model, eps=8/255, alpha=2/255, steps=50, eot_iter=2),                          # [Linf-W-2019]
        "28": OnePixel(model),                                                                      # [L0-B-2019]
        "29": FAB(model, eps=8/255, steps=100, n_restarts=1, targeted=False),          # [Linf/L2/L1-W-2020]
        # "NOISE" 
        "30": "Gaussian Noise",
        "31": "Shot Noise",
        "32": "Impulse Noise",
        "33": "Defocus Blur",
        "34": "Glass Blur",
        "35": "Motion Blur",
        "36": "Zoom Blur",
        "37": "Snow",
        "38": "Frost",
        "39": "Fog",
        "40": "Brightness",
        "41": "Contrast",
        "42": "Elastic",
        "43": "Pixelate",
        "44": "JPEG",
        "45": "Speckle Noise",
        "46": "Gaussian Blur",
        "47": "Saturate"
            }
    return ATKS

def get_attack_MNIST(classifier, model):
    ATKS = {
        # "ART"
        "0": SaliencyMapMethod(classifier=classifier, theta=0.1,gamma=1,batch_size=1,verbose=False),       # [L0-W-2015]
        "1": DeepFool(classifier=classifier, max_iter=100, verbose=False),                # [L2-W-2016]
        "2": UniversalPerturbation(classifier=classifier, verbose=False),   # [Linf-W-2017]
        "3": NewtonFool(classifier=classifier, verbose=False),              # [L0/L2-W-2017]
        "4": BoundaryAttack(estimator=classifier, max_iter=100, verbose=False),           # [L2-B-2018]
        "5": ElasticNet(classifier=classifier, verbose=False),              # [L1-W-2018]
        "6": ZooAttack(classifier=classifier,max_iter=30, learning_rate=0.1, abort_early= False, variable_h=0.1, verbose=False),   
        "7": SpatialTransformation(classifier=classifier, max_translation=20,max_rotation=20.0, verbose=False),   # [L?-B-2019]
        "8": HopSkipJump(classifier=classifier, verbose=False),             # [L2/Linf-B-2020]
        "9": SimBA(classifier=classifier),                   # [L2-B-2020]
        "10": ShadowAttack(estimator=classifier, lambda_tv=0.006,verbose=False),            # [L?-W-2020]
        "11": GeoDA(estimator=classifier, verbose=False),                                  # [Linf-B-2020]
        "12": Wasserstein(estimator=classifier,eps_step=0.01, max_iter=40,eps=0.5, conjugate_sinkhorn_max_iter=10,projected_sinkhorn_max_iter=10,verbose=False),                            # [L?-W-2020]
        # "TA"
        "13": FGSM(model, eps=60/255),                                                # [Linf-W-2014]
        "14": BIM(model, eps=35/255, alpha=2/255, steps=50),                          # [Linf-W-2017] steps=100
        "15": CW(model, c=2, lr=0.1, steps=50, kappa=0),                            # [Linf-W-2017]
        "16": MIFGSM(model, eps=35/255, alpha=2/255, steps=50, decay=0.1),            # [Linf-W-2018]
        "17": TIFGSM(model, eps=40/255, alpha=2/255, steps=50, decay=0.1),            # [Linf-W-2019]
        "18": PGD(model, eps=35/255, alpha=2/255, steps=50, random_start=True),       # [Linf-W-2019]
        "19": PGDL2(model, eps=3, alpha=0.2, steps=50),                              # [L2-W-2019]
        "20": TPGD(model, eps=35/255, alpha=2/255, steps=50),                         # [Linf-W-2019]
        "21": RFGSM(model, eps=35/255, alpha=2/255, steps=50),                        # [Linf-W-2020]
        "22": APGD(model, eps=35/255, steps=50, eot_iter=1, n_restarts=1, loss='ce'), # [Linf/L2-W-2020]
        "23": APGD(model, eps=35/255, steps=50, eot_iter=1, n_restarts=1, loss='dlr'),# [Linf/L2-W-2020]
        "24": FFGSM(model, eps=200/255, alpha=10/255),                                 # [Linf-W-2020]
        "25": Square(model, eps=40/255, n_queries=2500, n_restarts=1, loss='ce'),     # [Linf/L2-B-2020]
        "26": TIFGSM(model, eps=40/255, alpha=2/255, steps=50, diversity_prob=0.5, resize_rate=0.9), # [Linf-W-2019] steps=100
        "27": EOTPGD(model, eps=35/255, alpha=2/255, steps=50, eot_iter=2),                          # [Linf-W-2019]
        "28": OnePixel(model),                                                                      # [L0-B-2019]
        "29": FAB(model, eps=35/255, steps=100, n_restarts=1, targeted=False),          # [Linf/L2/L1-W-2020]
        # "NOISE" 
        "30": "Gaussian Noise",
        "31": "Shot Noise",
        "32": "Impulse Noise",
        "33": "Defocus Blur",
        "34": "Glass Blur",
        "35": "Motion Blur",
        "36": "Zoom Blur",
        "37": "Snow",
        "38": "Frost",
        "39": "Fog",
        "40": "Brightness",
        "41": "Contrast",
        "42": "Elastic",
        "43": "Pixelate",
        "44": "JPEG",
        "45": "Speckle Noise",
        "46": "Gaussian Blur",
        "47": "Saturate"
            }
    return ATKS


def get_attack_TinyImageNet(classifier, model):
    ATKS = {
        # "ART"
        "0": SaliencyMapMethod(classifier=classifier, theta=0.1,gamma=1,batch_size=1,verbose=False),       # [L0-W-2015]
        "1": DeepFool(classifier=classifier, max_iter=100, verbose=False),                # [L2-W-2016]
        "2": UniversalPerturbation(classifier=classifier, verbose=False),   # [Linf-W-2017]
        "3": NewtonFool(classifier=classifier, verbose=False),              # [L0/L2-W-2017]
        "4": BoundaryAttack(estimator=classifier, max_iter=100, verbose=False),           # [L2-B-2018]
        "5": ElasticNet(classifier=classifier, verbose=False),              # [L1-W-2018]
        "6": ZooAttack(classifier=classifier, max_iter=30, verbose=False),               # [L0-B-2018]
        "7": SpatialTransformation(classifier=classifier, max_translation=20,max_rotation=60.0, verbose=False),   # [L?-B-2019]
        "8": HopSkipJump(classifier=classifier, verbose=False),             # [L2/Linf-B-2020]
        "9": SimBA(classifier=classifier),                   # [L2-B-2020]
        "10": ShadowAttack(estimator=classifier, lambda_tv=0.006, verbose=False),            # [L?-W-2020]
        "11": GeoDA(estimator=classifier,verbose=False),                                  # [Linf-B-2020]
        "12": Wasserstein(estimator=classifier, regularization=20.0, max_iter=100, verbose=False),                            # [L?-W-2020]
        # "TA"
        "13": FGSM(model, eps=8/255),                                                # [Linf-W-2014]
        "14": BIM(model, eps=8/255, alpha=2/255, steps=50),                          # [Linf-W-2017] steps=100
        "15": CW(model, c=1, lr=0.01, steps=50, kappa=0),                            # [Linf-W-2017]
        "16": MIFGSM(model, eps=8/255, alpha=2/255, steps=50, decay=0.1),            # [Linf-W-2018]
        "17": TIFGSM(model, eps=8/255, alpha=2/255, steps=50, decay=0.1),            # [Linf-W-2019]
        "18": PGD(model, eps=8/255, alpha=2/225, steps=50, random_start=True),       # [Linf-W-2019]
        "19": PGDL2(model, eps=1, alpha=0.2, steps=50),                              # [L2-W-2019]
        "20": TPGD(model, eps=16/255, alpha=2/255, steps=50),                         # [Linf-W-2019]
        "21": RFGSM(model, eps=8/255, alpha=2/255, steps=50),                        # [Linf-W-2020]
        "22": APGD(model, eps=8/255, steps=50, eot_iter=1, n_restarts=1, loss='ce'), # [Linf/L2-W-2020]
        "23": APGD(model, eps=8/255, steps=50, eot_iter=1, n_restarts=1, loss='dlr'),# [Linf/L2-W-2020]
        "24": FFGSM(model, eps=8/255, alpha=10/255),                                 # [Linf-W-2020]
        "25": Square(model, eps=8/255, n_queries=5000, n_restarts=1, loss='ce'),     # [Linf/L2-B-2020]
        "26": TIFGSM(model, eps=8/255, alpha=2/255, steps=50, diversity_prob=0.5, resize_rate=0.9), # [Linf-W-2019] steps=100
        "27": EOTPGD(model, eps=8/255, alpha=2/255, steps=50, eot_iter=2),                          # [Linf-W-2019]
        "28": OnePixel(model),                                                                      # [L0-B-2019]
        "29": FAB(model, eps=8/255, steps=100, n_restarts=5, targeted=False),          # [Linf/L2/L1-W-2020]
        "30": FAB(model, eps=8/255, steps=100, n_restarts=5, targeted=False),          # [Linf/L2/L1-W-2020]
        # "NOISE" 
        "30": "Gaussian Noise",
        "31": "Shot Noise",
        "32": "Impulse Noise",
        "33": "Defocus Blur",
        "34": "Glass Blur",
        "35": "Motion Blur",
        "36": "Zoom Blur",
        "37": "Snow",
        "38": "Frost",
        "39": "Fog",
        "40": "Brightness",
        "41": "Contrast",
        "42": "Elastic",
        "43": "Pixelate",
        "44": "JPEG",
        "45": "Speckle Noise",
        "46": "Gaussian Blur",
        "47": "Saturate"
            }
    return ATKS

NOISE_PARAMS = {
    "Gaussian Noise": [0.04, 0.06, 0.08, 0.09, 0.10],
    "Shot Noise": [500, 250, 100, 75, 50],
    "Impulse Noise": [0.01, 0.02, 0.03, 0.05, 0.07],
    "Defocus Blur": [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (1, 0.2), (1.5, 0.1)],
    "Glass Blur": [(0.05, 1, 1), (0.25, 1, 1), (0.4, 1, 1), (0.25, 1, 2), (0.4, 1, 2)],
    "Motion Blur": [(6, 1), (6, 1.5), (6, 2), (8, 2), (9, 2.5)],
    "Zoom Blur": [
        np.arange(1, 1.06, 0.01),
        np.arange(1, 1.11, 0.01),
        np.arange(1, 1.16, 0.01),
        np.arange(1, 1.21, 0.01),
        np.arange(1, 1.26, 0.01),
    ],
    "Snow": [
        (0.1, 0.2, 1, 0.6, 8, 3, 0.95),
        (0.1, 0.2, 1, 0.5, 10, 4, 0.9),
        (0.15, 0.3, 1.75, 0.55, 10, 4, 0.9),
        (0.25, 0.3, 2.25, 0.6, 12, 6, 0.85),
        (0.3, 0.3, 1.25, 0.65, 14, 12, 0.8),
    ],
    "Frost": [(1, 0.2), (1, 0.3), (0.9, 0.4), (0.85, 0.4), (0.75, 0.45)],
    "Fog": [(0.2, 3), (0.5, 3), (0.75, 2.5), (1, 2), (1.5, 1.75)],
    "Brightness": [0.05, 0.1, 0.15, 0.2, 0.3],
    "Contrast": [0.75, 0.5, 0.4, 0.3, 0.15],
    "Elastic": [
        (0, 0, 0.08),
        (0.05, 0.2, 0.07),
        (0.08, 0.06, 0.06),
        (0.1, 0.04, 0.05),
        (0.1, 0.03, 0.03),
    ],
    "Pixelate": [0.95, 0.9, 0.85, 0.75, 0.65],
    "JPEG": [80, 65, 58, 50, 40],
    "Speckle Noise": [0.06, 0.1, 0.12, 0.16, 0.2],
    "Gaussian Blur": [0.4, 0.6, 0.7, 0.8, 1],
    "Spatter": [
        (0.62, 0.1, 0.7, 0.7, 0.5, 0),
        (0.65, 0.1, 0.8, 0.7, 0.5, 0),
        (0.65, 0.3, 1, 0.69, 0.5, 0),
        (0.65, 0.1, 0.7, 0.69, 0.6, 1),
        (0.65, 0.1, 0.5, 0.68, 0.6, 1),
    ],
    "Saturate": [(0.3, 0), (0.1, 0), (1.5, 0), (2, 0.1), (2.5, 0.2)],
}
