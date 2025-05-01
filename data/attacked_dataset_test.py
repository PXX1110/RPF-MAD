# Dataloader of Gidaris & Komodakis, CVPR 2018
# Adapted from:
# https://github.com/gidariss/FewShotWithoutForgetting/blob/master/dataloader.py
from __future__ import print_function

import os
import sys
sys.path.append(r'/home/users/pxx/workplace/5Adversarial/4Meta_AT_ResNet')
import os.path
import numpy as np
import random
import torch
import torch.utils.data as data
import torchnet as tnt
from PIL import Image
from src_sl.tools_meta.attack_common_tools import data_transform
from configs.attack_label_config import get_attack_train, get_attack_val, get_attack_test
from configs.experiment_config import CIFAR_CLASS_NAMES, MNIST_CLASS_NAMES, TINYIMAGENET_CLASS_NAMES, GTSRB_CLASS_NAMES

# Set the appropriate paths of the datasets here.
_ATTACKED_CIFAR_10_DATASET_DIR = r'/home/users/pxx/workplace/Datasets/MADS/MAD-C-S'
_ATTACKED_GTSRB_DATASET_DIR = r'/home/users/pxx/workplace/5Adversarial/4Meta_AT_ResNet/results_gtsrb/MAD-G_NEW'
_ATTACKED_MNIST_DATASET_DIR = r'/home/users/pxx/workplace/5Adversarial/MADS/MAD-M-R'
_ATTACKED_TinyImageNet_DATASET_DIR = r'/home/users/pxx/workplace/5Adversarial/4Meta_AT_ResNet/results/new_tinyimagenet'

def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)

    return label2inds

def load_data(files):
    data = {"data": [], 
            "labels": [], 
            "attack_labels":[]}
    for file in files:
        for i in os.listdir(file):
            path_img = ({"data": [os.path.join(file, i)], 
                        "labels": [int(os.path.basename(file))], 
                        "attack_labels": [int(os.path.split(os.path.split(os.path.split(file)[0])[0])[1])]})
            data = {key:[*data[key], *path_img[key]] for key in path_img}
    return data
    
def get_categorie(atk_train, opt, phase):
    train_class_dir = []
    for i in atk_train:
        if opt.dataset == 'CIFAR_10':
            if phase == "test":
                train_class_dir += [os.path.join(_ATTACKED_CIFAR_10_DATASET_DIR,'{}'.format(i),'{}'.format(phase))]
            else:
                train_class_dir += [os.path.join(_ATTACKED_CIFAR_10_DATASET_DIR,'{}'.format(i),'train')]
                train_class_dir += [os.path.join(_ATTACKED_CIFAR_10_DATASET_DIR,'{}'.format(i),'val')]
            class_names = CIFAR_CLASS_NAMES
        elif opt.dataset == 'TinyImageNet':
            if phase == "test":
                train_class_dir += [os.path.join(_ATTACKED_TinyImageNet_DATASET_DIR,'{}'.format(i),'{}'.format(phase))]
            else:
                train_class_dir += [os.path.join(_ATTACKED_TinyImageNet_DATASET_DIR,'{}'.format(i),'train')]
                train_class_dir += [os.path.join(_ATTACKED_TinyImageNet_DATASET_DIR,'{}'.format(i),'val')]
            class_names = TINYIMAGENET_CLASS_NAMES
        elif opt.dataset == 'MNIST':
            if phase == "test":
                train_class_dir += [os.path.join(_ATTACKED_MNIST_DATASET_DIR,'{}'.format(i),'{}'.format(phase))]
            else:
                train_class_dir += [os.path.join(_ATTACKED_MNIST_DATASET_DIR,'{}'.format(i),'train')]
                train_class_dir += [os.path.join(_ATTACKED_MNIST_DATASET_DIR,'{}'.format(i),'val')]
            class_names = MNIST_CLASS_NAMES
        elif opt.dataset == 'gtsrb':
            if phase == "test":
                train_class_dir += [os.path.join(_ATTACKED_GTSRB_DATASET_DIR,'{}'.format(i),'{}'.format(phase))]
            else:
                train_class_dir += [os.path.join(_ATTACKED_GTSRB_DATASET_DIR,'{}'.format(i),'train')]
                train_class_dir += [os.path.join(_ATTACKED_GTSRB_DATASET_DIR,'{}'.format(i),'val')]
            class_names = GTSRB_CLASS_NAMES
    class_dir = []
    for m in train_class_dir:   
        for n in os.listdir(train_class_dir[0]):
            class_dir += [os.path.join(m,n)]
    return class_dir, class_names
    
class ATTACKED_DATASET_TEST(data.Dataset):
    def __init__(self, opt, phase='train', do_not_use_random_transform=False, ):

        assert(phase=='train' or phase=='val' or phase=='test')
        self.opt = opt
        self.phase = phase
        self.atk_train = get_attack_train(self.opt.attack, self.opt.dataset)
        self.atk_val = get_attack_val(self.opt.attack, self.opt.dataset)
        self.atk_test = get_attack_test(self.opt.attack, self.opt.dataset)
        
        print('Loading Attack_dataset - phase {0}'.format(phase))
        if self.phase == "train":
            file_test_categories_train, class_names = get_categorie(self.atk_train, opt, "train")
            file_test_categories_val, class_names = get_categorie(self.atk_train, opt, "test")
        elif self.phase == "val":  
            file_test_categories_train, class_names = get_categorie(self.atk_val, opt, "train")
            file_test_categories_val, class_names = get_categorie(self.atk_val, opt, "test")
        elif self.phase == "test":
            file_test_categories_train, class_names = get_categorie(self.atk_test, opt, "train")
            file_test_categories_val, class_names = get_categorie(self.atk_test, opt, "test")

        data_base = load_data(file_test_categories_train)
        data_novel = load_data(file_test_categories_val)
        self.data = {"base": data_base['data'], "novel":data_novel['data']}
        self.labels = {"base": data_base['labels'], "novel":data_novel['labels']}
        self.attack_labels = {"base": data_base['attack_labels'], "novel":data_novel['attack_labels']}
        self.label2ind_base = buildLabelIndex(data_base['labels'])
        self.label2ind_novel = buildLabelIndex(data_novel['labels'])
        self.attack_label2ind_base = buildLabelIndex(data_base['attack_labels'])
        self.attack_label2ind_novel = buildLabelIndex(data_novel['attack_labels'])
        self.attack_labelIds = sorted(self.attack_label2ind_base.keys())
        self.num_cats = len(class_names)                                             
        self.attack_num_cats = len(self.attack_labelIds)

    def __getitem__(self, cat_set, index):

        fn, label = self.data[cat_set][index], self.labels[cat_set][index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(fn).convert('RGB')
        # if self.transform is not None:
            # img = self.transform(img)
        img = data_transform(self.opt, img, "test")
        return img, label

    def __len__(self):
        return len(self.data)


class FewShotDataloader_Test():
    def __init__(self,
                 dataset,
                 attack_ID,
                 nExemplars=1, # number of training examples per novel category.
                 nTestNovel=1 * 10 * 6, # number of test examples for all the novel categories.
                 nTestBase=2 * 10 * 15, # number of test examples for all the base categories.
                 batch_size=1, # number of training episodes per batch.
                 num_workers=0,
                 epoch_size=100, # number of batches per epoch.
                 mix=True
                 ):
        self.dataset = dataset
        self.attack_ID = attack_ID
        self.phase = self.dataset.phase
        self.nExemplars = nExemplars                                                     # 15
        self.nTestNovel = nTestNovel                                                     # 1 * 10 * 6
        self.nTestBase = nTestBase                                                       # 2 * 10 * 15
        self.batch_size = batch_size                                                     # 8/1           
        self.epoch_size = epoch_size                                                     # 8 * 10  
        self.num_workers = num_workers
        self.is_eval_mode = (self.phase=='test') or (self.phase=='val')

    def sampleImageIdsFrom(self, cat_set, cat_id, sample_size=1):
        """
        Samples `sample_size` number of unique image ids picked from the
        category `cat_id` (i.e., self.dataset.label2ind[cat_id]).

        Args:
            cat_id: a scalar with the id of the category from which images will
                be sampled.
            sample_size: number of images that will be sampled.

        Returns:
            image_ids: a list of length `sample_size` with unique image ids.
        """
        # assert(cat_id in self.dataset.attack_label2ind)
        # assert(len(self.dataset.attack_label2ind[cat_id]) >= sample_size)
        if cat_set=='base':
            img_id = np.empty(shape=(0,sample_size))
            for i in range(len(self.dataset.label2ind_base)):
                label = [x for x in self.dataset.label2ind_base[i] if x in self.dataset.attack_label2ind_base[cat_id]]
                img_id = np.vstack((img_id,np.array((random.sample(label, sample_size))))).astype(int)
        else:
            img_id = np.empty(shape=(0,sample_size))
            for i in range(len(self.dataset.label2ind_novel)):
                label = [x for x in self.dataset.label2ind_novel[i] if x in self.dataset.attack_label2ind_novel[cat_id]]
                img_id = np.vstack((img_id,np.array((random.sample(label, sample_size))))).astype(int)
        # Note: random.sample samples elements without replacement.
        return img_id

    def sample_Attack_Categories(self, sample_size=1):
        """
        Samples `sample_size` number of unique categories picked from the
        `cat_set` set of categories. `cat_set` can be either 'base' or 'novel'.

        Args:
            cat_set: string that specifies the set of categories from which
                categories will be sampled.
            sample_size: number of categories that will be sampled.

        Returns:
            cat_ids: a list of length `sample_size` with unique category ids.
        """
        labelIds = self.dataset.attack_labelIds

        assert(len(labelIds) >= sample_size)
        # return sample_size unique categories chosen from labelIds set of
        # categories (that can be either self.labelIds_base or self.labelIds_novel)
        # Note: random.sample samples elements without replacement.
        return list(labelIds)[sample_size]

    def sample_base_and_novel_categories(self):
        """
        Samples `nKbase` number of base categories and `nKnovel` number of novel
        categories.

        Args:
            nKbase: number of base categories
            nKnovel: number of novel categories

        Returns:
            Kbase: a list of length 'nKbase' with the ids of the sampled base
                categories.
            Knovel: a list of lenght 'nKnovel' with the ids of the sampled novel
                categories.
        """
        Knovel = self.sample_Attack_Categories(self.attack_ID)

        return Knovel

    def sample_train_and_test_examples_for_attacks(
            self, Knovel, nTestNovel, nExemplars):
        """Samples train and test examples of the novel categories.

        Args:
    	    Knovel: a list with the ids of the novel categories.
            nTestNovel: the total number of test images that will be sampled
                from all the novel categories.
            nExemplars: the number of training examples per novel category that
                will be sampled.
            nKbase: the number of base categories. It is used as offset of the
                category index of each sampled image.

        Returns:
            Tnovel: a list of length `nTestNovel` with 2-element tuples. The
                1st element of each tuple is the image id that was sampled and
                the 2nd element is its category label (which is in the range
                [nKbase, nKbase + len(Knovel) - 1]).
            Exemplars: a list of length len(Knovel) * nExemplars of 2-element
                tuples. The 1st element of each tuple is the image id that was
                sampled and the 2nd element is its category label (which is in
                the ragne [nKbase, nKbase + len(Knovel) - 1]).
        """
        Tnovel = []
        Exemplars = []
        nEvalExamplesPerClass = int(nTestNovel / self.dataset.num_cats)  #   1
        imd_ids_base = self.sampleImageIdsFrom("base", Knovel, sample_size=nExemplars)
        imd_ids_novel = self.sampleImageIdsFrom("novel", Knovel, sample_size=nEvalExamplesPerClass)
        imds_ememplars = (imd_ids_base.T.reshape(-1)).tolist()
        imds_tnovel = (imd_ids_novel.T.reshape(-1)).tolist()
        Exemplars += [(img_id, Knovel) for img_id in imds_ememplars]
        Tnovel += [(img_id, Knovel) for img_id in imds_tnovel]  

        return Tnovel, Exemplars

    def sample_episode(self):
        """Samples a training episode."""    
        nTestNovel = self.nTestNovel
        nExemplars = self.nExemplars
        Knovel = self.sample_base_and_novel_categories()
        Tnovel, Exemplars_novel = self.sample_train_and_test_examples_for_attacks(
            Knovel, nTestNovel, nExemplars)
        # concatenate the base and novel category examples.
        Test = Tnovel
        Exemplars = Exemplars_novel
        random.shuffle(Test)
        random.shuffle(Exemplars)
        
        return Exemplars, Test

    def createExamplesTensorData(self, examples,cat_set):
        """
        Creates the examples image and label tensor data.
        Args:
            examples: a list of 2-element tuples, each representing a
                train or test example. The 1st element of each tuple
                is the image id of the example and 2nd element is the
                category label of the example, which is in the range
                [0, nK - 1], where nK is the total number of categories
                (both novel and base).
        Returns:
            images: a tensor of shape [nExamples, Height, Width, 3] with the
                example images, where nExamples is the number of examples
                (i.e., nExamples = len(examples)).
            labels: a tensor of shape [nExamples] with the category label
                of each example.
        """
        data = [self.dataset.__getitem__(cat_set, img_idx) for img_idx, _ in examples]
        images = torch.stack([data[i][0] for i in range(len(data))], dim=0)
        labels = torch.LongTensor([data[i][1] for i in range(len(data))])

        return images, labels

    def get_iterator(self, epoch=2):
        rand_seed = epoch
        random.seed(rand_seed)
        np.random.seed(rand_seed)
        def load_function(iter_idx):
            Exemplars, Test= self.sample_episode()
            Xt, Yt = self.createExamplesTensorData(Test,"novel")
            if len(Exemplars) > 0:
                Xe, Ye = self.createExamplesTensorData(Exemplars,"base")
                return Xe, Ye, Xt, Yt
            else:
                return Xt, Yt
        tnt_dataset = tnt.dataset.ListDataset(
            elem_list=range(self.epoch_size), load=load_function)
        data_loader = tnt_dataset.parallel(
            batch_size=self.batch_size,
            num_workers=(0 if self.is_eval_mode else self.num_workers),
            shuffle=(False if self.is_eval_mode else True),
            drop_last=True)
        return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return int(self.epoch_size / self.batch_size)
