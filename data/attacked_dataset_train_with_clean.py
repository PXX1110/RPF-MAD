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
from configs.experiment_config import CIFAR_CLASS_NAMES, MNIST_CLASS_NAMES, TINYIMAGENET_CLASS_NAMES 

# Set the appropriate paths of the datasets here.
_ATTACKED_CIFAR_10_DATASET_DIR = r'/home/users/pxx/workplace/Datasets/MADS/MAD-C'
_ATTACKED_MNIST_DATASET_DIR = r'/home/users/pxx/workplace/5Adversarial/4Meta_AT_ResNet/results/new_right_mnist_resnet'
_ATTACKED_TinyImageNet_DATASET_DIR = r'/home/users/pxx/workplace/5Adversarial/4Meta_AT_ResNet/results/new_right_tiny_imageNet'

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
    
def get_categorie(atk_train, opt):
    train_class_dir = []
    for i in atk_train:
        if opt.dataset == 'CIFAR_10':
            train_class_dir += [os.path.join(_ATTACKED_CIFAR_10_DATASET_DIR,'{}'.format(i),'train')]
            train_class_dir += [os.path.join(_ATTACKED_CIFAR_10_DATASET_DIR,'{}'.format(i),'val')]
            class_names = CIFAR_CLASS_NAMES
        elif opt.dataset == 'TinyImageNet':
            train_class_dir += [os.path.join(_ATTACKED_TinyImageNet_DATASET_DIR,'{}'.format(i),'train')]
            train_class_dir += [os.path.join(_ATTACKED_TinyImageNet_DATASET_DIR,'{}'.format(i),'val')]
            class_names = TINYIMAGENET_CLASS_NAMES
        elif opt.dataset == 'MNIST':
            train_class_dir += [os.path.join(_ATTACKED_MNIST_DATASET_DIR,'{}'.format(i),'train')]
            train_class_dir += [os.path.join(_ATTACKED_MNIST_DATASET_DIR,'{}'.format(i),'val')]
            class_names = MNIST_CLASS_NAMES
    class_dir = []
    for m in train_class_dir:   
        for n in os.listdir(train_class_dir[0]):
            class_dir += [os.path.join(m,n)]
    return class_dir, class_names
    
class ATTACKED_DATASET(data.Dataset):
    def __init__(self, opt, phase='train', do_not_use_random_transf=False, ):

        assert(phase=='train' or phase=='val' or phase=='test')
        self.opt = opt
        self.phase = phase
        self.atk_train = get_attack_train(self.opt.attack, self.opt.dataset)
        self.atk_val = get_attack_val(self.opt.attack, self.opt.dataset)
        self.atk_test = get_attack_test(self.opt.attack, self.opt.dataset)
        print('Loading Attack_dataset - phase {0}'.format(phase))
        file_train_categories, class_names = get_categorie(self.atk_train, opt)
        file_val_categories, class_names = get_categorie(self.atk_val, opt)
        file_test_categories, class_names = get_categorie(self.atk_test, opt)
        
        if self.phase=='train':
            # During training phase we only load the training phase images
            # of the training categories (aka base categories).
            data_train = load_data(file_train_categories)
            self.data = data_train['data']
            self.labels = data_train['labels']
            self.attack_labels = data_train['attack_labels']
            self.label2ind = buildLabelIndex(self.labels)
            self.attack_label2ind = buildLabelIndex(self.attack_labels)
            self.attack_labelIds = sorted(self.attack_label2ind.keys())
            self.num_cats = len(class_names)                            
            self.attack_num_cats = len(self.attack_labelIds)
            self.attack_labelIds_base = self.attack_labelIds
            self.attack_num_cats_base = len(self.attack_labelIds_base)
        elif self.phase=='val' or self.phase=='test':
            if self.phase=='test':
                data_base = load_data(file_train_categories)
                data_novel = load_data(file_test_categories)
            else: # phase=='val'
                data_base = load_data(file_train_categories)
                data_novel = load_data(file_val_categories)
            self.data = np.concatenate(
                [data_base['data'], data_novel['data']], axis=0)
            self.labels = data_base['labels'] + data_novel['labels']
            self.attack_labels = data_base['attack_labels'] + data_novel['attack_labels']
            self.label2ind = buildLabelIndex(self.labels)
            self.attack_label2ind = buildLabelIndex(self.attack_labels)
            self.attack_labelIds = sorted(self.attack_label2ind.keys())
            self.num_cats = len(class_names)                                             
            self.attack_num_cats = len(self.attack_labelIds)
            self.attack_labelIds_base = buildLabelIndex(data_base['attack_labels']).keys()
            self.attack_labelIds_novel = buildLabelIndex(data_novel['attack_labels']).keys()
            self.attack_num_cats_base = len(self.attack_labelIds_base)
            self.attack_num_cats_novel = len(self.attack_labelIds_novel)
            # intersection = set(self.labelIds_base) & set(self.labelIds_novel)   # intersection() 方法用于返回两个或更多集合中都包含的元素，即交集。
            # assert(len(intersection) == 0)
        else:
            raise ValueError('Not valid phase {0}'.format(self.phase))
            
    def __getitem__(self, index):
        fn, label = self.data[index], self.labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(fn).convert('RGB')
        # if self.transform is not None:
            # img = self.transform(img)
        img = data_transform(self.opt, img, self.phase)
        return img, label, fn

    def __len__(self):
        return len(self.data)


class FewShotDataloader():
    def __init__(self,
                 adv_dataset,
                 clean_dataset,
                 nKnovel=1, # number of novel categories.
                 nKbase=2, # number of base categories.
                 nExemplars=1, # number of training examples per novel category.
                 nTestNovel=1 * 10 * 6, # number of test examples for all the novel categories.
                 nTestBase=2 * 10 * 15, # number of test examples for all the base categories.
                 batch_size=1, # number of training episodes per batch.
                 num_workers=0,
                 epoch_size=100, # number of batches per epoch.
                 mix=True
                 ):
        self.adv_dataset = adv_dataset
        self.clean_dataset = clean_dataset
        self.phase = self.adv_dataset.phase
        max_possible_nKnovel = (self.adv_dataset.attack_num_cats_base if self.phase=='train'
                                else self.adv_dataset.attack_num_cats_novel)
        assert(nKnovel >= 0 and nKnovel < max_possible_nKnovel)
        self.nKnovel = nKnovel                                                           # 1
        self.mix = mix
        max_possible_nKbase = self.adv_dataset.attack_num_cats_base
        nKbase = nKbase if nKbase >= 0 else max_possible_nKbase
        # if self.phase=='train' and nKbase > 0:
        if nKbase > 0:
            nKbase -= self.nKnovel
            max_possible_nKbase -= self.nKnovel
        assert(nKbase >= 0 and nKbase <= max_possible_nKbase)
        self.nKbase = nKbase                                                             # 2-1
        self.nExemplars = nExemplars                                                     # 15
        self.nTestNovel = nTestNovel                                                     # 1 * 10 * 6
        self.nTestBase = nTestBase                                                       # 2 * 10 * 15
        self.batch_size = batch_size                                                     # 8/1           
        self.epoch_size = epoch_size                                                     # 8 * 10  
        self.num_workers = num_workers
        self.is_eval_mode = (self.phase=='test') or (self.phase=='val')

    def sampleImageIdsFrom(self, cat_id, sample_size=1):
        """
        Samples `sample_size` number of unique image ids picked from the
        category `cat_id` (i.e., self.adv_dataset.label2ind[cat_id]).

        Args:
            cat_id: a scalar with the id of the category from which images will
                be sampled.
            sample_size: number of images that will be sampled.

        Returns:
            image_ids: a list of length `sample_size` with unique image ids.
        """
        assert(cat_id in self.adv_dataset.attack_label2ind)
        # assert(len(self.adv_dataset.attack_label2ind[cat_id]) >= sample_size)
        img_id = np.empty(shape=(0,sample_size))
        for i in range(len(self.adv_dataset.label2ind)):
            label = [x for x in self.adv_dataset.label2ind[i] if x in self.adv_dataset.attack_label2ind[cat_id]]
            img_id = np.vstack((img_id, np.array((random.sample(label, sample_size))))).astype(int)
        # Note: random.sample samples elements without replacement.
        return img_id

    def sample_Attack_Categories(self, cat_set, sample_size=1):
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
        if cat_set=='base':
            labelIds = self.adv_dataset.attack_labelIds_base
        elif cat_set=='novel':
            labelIds = self.adv_dataset.attack_labelIds_novel
        else:
            raise ValueError('Not recognized category set {}'.format(cat_set))

        assert(len(labelIds) >= sample_size)
        # return sample_size unique categories chosen from labelIds set of
        # categories (that can be either self.labelIds_base or self.labelIds_novel)
        # Note: random.sample samples elements without replacement.
        return random.sample(labelIds, sample_size)

    def sample_base_and_novel_categories(self, nKbase, nKnovel):
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
        if self.is_eval_mode:
            if self.mix == True:
                assert(nKnovel <= self.adv_dataset.attack_num_cats_novel)
                # sample from the set of base categories 'nKbase' number of base
                # categories.
                cats_ids = self.sample_Attack_Categories('base', nKbase) + self.sample_Attack_Categories('novel', nKnovel)
                # sample from the set of novel categories 'nKnovel' number of novel
                # categories.
                random.shuffle(cats_ids)
                Knovel = sorted(cats_ids[:nKnovel])
                Kbase = sorted(cats_ids[nKnovel:])
            else:
                # sample from the set of novel categories 'nKnovel' number of novel
                # categories.
                Knovel = sorted(self.sample_Attack_Categories('novel', nKnovel))
                Kbase = sorted(self.sample_Attack_Categories('base', nKbase))
        else:
            # sample from the set of base categories 'nKnovel' + 'nKbase' number
            # of categories.
            cats_ids = self.sample_Attack_Categories('base', nKnovel+nKbase)
            assert(len(cats_ids) == (nKnovel+nKbase))
            # Randomly pick 'nKnovel' number of fake novel categories and keep
            # the rest as base categories.
            random.shuffle(cats_ids)
            Knovel = sorted(cats_ids[:nKnovel])
            Kbase = sorted(cats_ids[nKnovel:])
        return Kbase, Knovel

    def sample_train_and_test_examples_for_attacks(
            self, Knovel, nTestNovel, nExemplars, nKbase):
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
        if len(Knovel) == 0:
            return [], []
        nKnovel = len(Knovel)                                            #   2
        Tnovel = []
        Exemplars = []
        assert((nTestNovel % (nKnovel+1)) == 0)
        nEvalExamplesPerClass = int(nTestNovel / self.adv_dataset.num_cats)  #   1
        for Knovel_idx in range(len(Knovel)):
            imd_ids = self.sampleImageIdsFrom(
                Knovel[Knovel_idx],
                sample_size=(nEvalExamplesPerClass + nExemplars))
            imds_ememplars = (imd_ids.T[nEvalExamplesPerClass:].reshape(-1)).tolist()
            imds_tnovel = (imd_ids.T[:nEvalExamplesPerClass].reshape(-1)).tolist()
            Exemplars += [(img_id, Knovel_idx) for img_id in imds_ememplars]
            Tnovel += [(img_id, Knovel_idx) for img_id in imds_tnovel]  
        # assert(len(Tnovel) == nTestNovel)
        assert(len(Exemplars) == len(Knovel) * nExemplars * self.adv_dataset.num_cats)
        return Tnovel, Exemplars

    def sample_episode(self):
        """Samples a training episode."""
        nKnovel = self.nKnovel        # 1
        nKbase = self.nKbase          # 1        
        nTestNovel = self.nTestNovel
        nTestBase = self.nTestBase
        nExemplars = self.nExemplars
        Kbase, Knovel = self.sample_base_and_novel_categories(nKbase, nKnovel)
        Tbase, Exemplars_base = self.sample_train_and_test_examples_for_attacks(
            Kbase, nTestNovel, nExemplars, nKbase)
        Tnovel, Exemplars_novel = self.sample_train_and_test_examples_for_attacks(
            Knovel, nTestNovel, nExemplars, nKnovel)
        # concatenate the base and novel category examples.
        Test = Tnovel
        Exemplars = Exemplars_base + Exemplars_novel
        random.shuffle(Test)
        random.shuffle(Exemplars)
        Kall = Kbase + Knovel
        return Exemplars, Test, Kall, nKbase

    def createExamplesTensorData(self, examples):
        """
        Creates the examples image and label tensor data.
        Args:
            examples: a list of 2-element tuples, each representing a
                train or test example. The 1st element of each tuple
                is the image id of the example and 2nd element is the
                category label of the example, which is in the range
                [0, nK - 1], where nK is the total number of categories
                (both novel and base).
            clean_examples: Optional clean dataset to load clean images.

        Returns:
            images: a tensor of shape [nExamples, Height, Width, 3] with the
                example images.
            clean_images: Same as images but for clean data if provided.
            labels: a tensor of shape [nExamples] with the category label
                of each example.
        """
        adv_images_list = []
        adv_labels_list = []
        clean_images_list = []  # For clean examples
        clean_labels_list = []  # For clean examples
        # Loop through examples and optionally clean examples
        for idx, example in enumerate(examples):
            img_idx = example[0]
            label = example[1]
            
            # Fetch data from dataset
            adv_img_data = self.adv_dataset[img_idx]
            adv_images_list.append(adv_img_data[0])  # Append the adv image
            adv_labels_list.append(adv_img_data[1])  # Append the adv label
            adv_img_name = adv_img_data[2]
            # If clean_examples is provided, fetch corresponding clean data
            if self.clean_dataset is not None:
                clean_img_data = self.clean_dataset[os.path.basename(adv_img_name)]  # Fetch clean image
                clean_images_list.append(clean_img_data[0])  # Append the clean image
                clean_labels_list.append(clean_img_data[1])  # Append the clean label

            assert adv_img_data[1] == clean_img_data[1], "Mismatch between clean and adversarial labels at index {}".format(idx)
        # Convert list of images and labels to tensors
        adv_images = torch.stack(adv_images_list, dim=0)
        adv_labels = torch.LongTensor(adv_labels_list)

        # If clean_examples is provided, return clean_images as well
        if self.clean_dataset is not None:
            clean_images = torch.stack(clean_images_list, dim=0)
            clean_labels = torch.LongTensor(clean_labels_list)

            return adv_images, adv_labels, clean_images
        else:
            return adv_images, adv_labels

    def get_iterator(self, epoch=0):
        rand_seed = epoch
        random.seed(rand_seed)
        np.random.seed(rand_seed)
        def load_function(iter_idx):
            Exemplars, Test, Kall, nKbase = self.sample_episode()
            if self.clean_dataset is None:
                Xt, Yt = self.createExamplesTensorData(Test)
            else:
                Xt, Yt, Xtc = self.createExamplesTensorData(Test)
            Kall = torch.LongTensor(Kall)
            if len(Exemplars) > 0:
                if self.clean_dataset is None:
                    Xe, Ye = self.createExamplesTensorData(Exemplars)
                    return Xe, Ye, Xt, Yt, Kall, nKbase
                else:
                    Xe, Ye, Xec = self.createExamplesTensorData(Exemplars)
                    return Xe, Ye, Xec, Xt, Yt, Xtc, Kall, nKbase
            else:
                return Xt, Yt, Kall, nKbase
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
