import sys
sys.path.append(r'/home/users/pxx/workplace/5Adversarial/4Meta_AT_ResNet')

def get_attack_train(optattack, optattack_dataset):
    if optattack_dataset == "CIFAR_10":
        atk_train1 = [
            # "ART"     6
            1,
            3,
            4,5,
            6,
            7,
            # "TA"      12
            13,14,15,16,
            17,
            18,
            19,
            20,
            21,22,
            23,
            24
            ]
        atk_train2 = [
            # "NOISE"   8
            30,31,34,36,37,39,40,41]
    elif optattack_dataset == "MNIST":
        atk_train1 = [
            # "ART"     3
            # 1,2,
            3,4,5,
            # 7,
            # "TA"      11
            13,14,15,16,17,18,19,20,21,22,23
            # ,24
            ]
        atk_train2 = [
            # "NOISE"   12
            30,31,32,33,34,35,36,37,38,39,40,41]
    elif optattack_dataset == "TinyImageNet":
        atk_train1 = [
            # "ART"     6
            1,2,3,4,5,6,7,
            # "TA"      11
            13,14,15,16,17,18,19,20,21,22,23
            ]
        atk_train2 = [
            # "NOISE"   12
            30,31,32,33,34,35,36,37,38,39,40,41]
    elif optattack_dataset == "gtsrb":
        atk_train1 = [
            # "ART"     6
            1,3,
            4,5,6,7,
            # "TA"      12
            13,14,15,16,17,18,19,20,21,22,23,24
            ]
        atk_train2 = [
            # "NOISE"   8
            30,31,34,36,37,39,40,41]
        
    if optattack[0] ==True and optattack[1] ==False:
        atk_train = atk_train1
    elif optattack[0] ==True and optattack[1] ==True:
        atk_train = atk_train1 + atk_train2

    return atk_train

def get_attack_val(optattack, optattack_dataset):
    if optattack_dataset == "CIFAR_10":
        atk_val1 = [
            # "ART"   2
            # 8,
            9,
            # "TA"    2
            25,26]
        atk_val2 = [
            # "NOISE" 2
            43,44]
    elif optattack_dataset == "MNIST":
        atk_val1 = [
            # "ART"   2
            6, 8,
            # 9,
            # "TA"    1
            25]
        atk_val2 = [
            # "NOISE" 2
            42,43]
    elif optattack_dataset == "TinyImageNet":
        atk_val1 = [
            # "ART"   2
            8,9,
            # "TA"    2
            25,26]
        atk_val2 = [
            # "NOISE" 2
            42,43]
    if optattack_dataset == "gtsrb":
        atk_val1 = [
            # "ART"   2
            8,9,
            # "TA"    2
            25,26]
        atk_val2 = [
            # "NOISE" 2
            43,44]
        
    if optattack[0] ==True and optattack[1] ==False:
        atk_val = atk_val1
    elif optattack[0] ==True and optattack[1] ==True:
        atk_val = atk_val1 + atk_val2 

    return atk_val

def get_attack_test(optattack, optattack_dataset):
    if optattack_dataset == "CIFAR_10": 
        atk_test1 = [
            # "ART"     3
            2,
            10,11,12,
            # "TA"      3
            27,28,29
            ]
        atk_test2 = [
            # "NOISE"   4       
            45,46,47]
    elif optattack_dataset == "MNIST":
        atk_test1 = [
            # "ART"     2
            7, 11,
            # "TA"      3
            # 24,
            26, 27, 29]
        atk_test2 = [
            # "NOISE"   4       
            44,45,46,47]
    elif optattack_dataset == "TinyImageNet":
        atk_test1 = [
            # "ART"     2
            10,11,
            # "TA"      2
            24,27]
        atk_test2 = [
            # "NOISE"   4       
            44,45,46,47]
    elif optattack_dataset == "gtsrb":
        atk_test1 = [
            # "ART"     2
            # 10,11,
            # "TA"      2
            2,10,11,
            27,28,29
            ]
        atk_test2 = [
            # "NOISE"   4       
            44,45,46,47]
        
    if optattack[0] ==True and optattack[1] ==False:
        atk_test = atk_test1
    elif optattack[0] ==True and optattack[1] ==True:
        atk_test = atk_test1 + atk_test2 

    return atk_test