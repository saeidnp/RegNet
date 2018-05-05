"""
Create train, valid, test iterators for CIFAR-100 [1].
Easily extended to MNIST, CIFAR-10 and Imagenet.

[1]: https://discuss.pytorch.org/t/feedback-on-pytorch-for-kaggle-competitions/2252/4
"""

import torch
import numpy as np
import random

from utils import *
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


def getTrainValidLoader(data_dir,
                        batch_size,
                        augment,
                        random_seed,
                        valid_size=0.1,
                        shuffle=True,
                        show_sample=False,
                        num_workers=4,
                        pin_memory=False,
                        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5]),
                        portion_to_keep = [1.0]*100
                        ):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-100 dataset. A sample
    9x9 grid of the images can be optionally displayed.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """

    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    dl_flag = not doesFileExist("./data/cifar-100-python.tar.gz")

    #normalize = transforms.Normalize(
    #    mean=[0.485, 0.456, 0.406],
    #    std=[0.229, 0.224, 0.225],
    #)

    # define transforms
    valid_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
    ])
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    # load the dataset
    train_dataset = datasets.CIFAR100(
        root=data_dir, train=True,
        download=dl_flag, transform=train_transform,
    )

    valid_dataset = datasets.CIFAR100(
        root=data_dir, train=True,
        download=dl_flag, transform=valid_transform,
    )

    # One_time preprocess to build label to image indices dictionary
    num_of_img_per_label = 500
    label_imgidx_dict = {}

    for img_idx, data in enumerate(train_dataset):
        # get the inputs
        img, label = data
        if label in label_imgidx_dict:
            label_imgidx_dict[label].append(img_idx)
        else:
            label_imgidx_dict[label] = []
            label_imgidx_dict[label].append(img_idx)

    num_train = len(train_dataset)
    #indices = list(range(num_train))
    indices = customSampler(label_imgidx_dict, portion_to_keep)
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    # visualize some images
    if show_sample:
        sample_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=9, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        data_iter = iter(sample_loader)
        images, labels = data_iter.next()
        X = images.numpy().transpose([0, 2, 3, 1])
        plot_images(X, labels)

    return (train_loader, valid_loader)


def getTestLoader(data_dir,
                    batch_size,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=False,
                    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
                    ):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-100 dataset.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - data_loader: test set iterator.
    """

    dl_flag = not doesFileExist("./data/cifar-100-python.tar.gz")

    #normalize = transforms.Normalize(
    #    mean=[0.485, 0.456, 0.406],
    #    std=[0.229, 0.224, 0.225],
    #)

    # define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    dataset = datasets.CIFAR100(
        root=data_dir, train=False,
        download=dl_flag, transform=transform,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader

def getClasses(data_dir):
    meta_pickle = unpickle(data_dir + 'cifar-100-python/meta')
    classes = meta_pickle[b'fine_label_names']
    return classes

def getSuperClasses(data_dir):
    meta_pickle = unpickle(data_dir + 'cifar-100-python/meta')
    super_classes = meta_pickle[b'coarse_label_names']
    return super_classes

def getClassNamesDict(data_dir):
    classes = getClasses(data_dir)
    super_classes = getSuperClasses(data_dir)
    class_name_to_class_id = {classes[i].decode("utf-8"): i for i, c in enumerate(classes)}
    return class_name_to_class_id

def getParentList(data_dir):
    ## Implement the fixed hierarchy:
    classes = getClasses(data_dir)
    super_classes = getSuperClasses(data_dir)
    class_name_to_class_id = {classes[i].decode("utf-8"): i for i, c in enumerate(classes)}
    superclass_name_to_superclass_id = {super_classes[i].decode("utf-8"): i for i, c in enumerate(super_classes)}
    class_name_to_superclass_name = {"beaver": "aquatic_mammals", "dolphin": "aquatic_mammals", "otter": "aquatic_mammals",
               "seal": "aquatic_mammals", "whale": "aquatic_mammals", "aquarium_fish": "fish",
               "flatfish": "fish", "ray": "fish", "shark": "fish", "trout": "fish", "orchid": "flowers",
               "poppy": "flowers", "rose": "flowers", "sunflower": "flowers", "tulip": "flowers",
               "bottle": "food_containers", "bowl": "food_containers", "can": "food_containers",
               "cup": "food_containers", "plate": "food_containers", "apple": "fruit_and_vegetables",
               "mushroom": "fruit_and_vegetables", "orange": "fruit_and_vegetables", "pear": "fruit_and_vegetables",
               "sweet_pepper": "fruit_and_vegetables", "clock": "household_electrical_devices",
               "keyboard": "household_electrical_devices", "lamp": "household_electrical_devices",
               "telephone": "household_electrical_devices", "television": "household_electrical_devices",
               "bed": "household_furniture", "chair": "household_furniture", "couch": "household_furniture",
               "table": "household_furniture", "wardrobe": "household_furniture", "bee": "insects",
               "beetle": "insects", "butterfly": "insects", "caterpillar": "insects", "cockroach": "insects",
               "bear": "large_carnivores", "leopard": "large_carnivores", "lion": "large_carnivores",
               "tiger": "large_carnivores", "wolf": "large_carnivores", "bridge": "large_man-made_outdoor_things",
               "castle": "large_man-made_outdoor_things", "house": "large_man-made_outdoor_things",
               "road": "large_man-made_outdoor_things", "skyscraper": "large_man-made_outdoor_things",
               "cloud": "large_natural_outdoor_scenes", "forest": "large_natural_outdoor_scenes",
               "mountain": "large_natural_outdoor_scenes", "plain": "large_natural_outdoor_scenes",
               "sea": "large_natural_outdoor_scenes", "camel": "large_omnivores_and_herbivores",
               "cattle": "large_omnivores_and_herbivores", "chimpanzee": "large_omnivores_and_herbivores",
               "elephant": "large_omnivores_and_herbivores", "kangaroo": "large_omnivores_and_herbivores",
               "fox": "medium_mammals", "porcupine": "medium_mammals", "possum": "medium_mammals",
               "raccoon": "medium_mammals", "skunk": "medium_mammals", "crab": "non-insect_invertebrates",
               "lobster": "non-insect_invertebrates", "snail": "non-insect_invertebrates",
               "spider": "non-insect_invertebrates", "worm": "non-insect_invertebrates", "baby": "people",
               "boy": "people", "girl": "people", "man": "people", "woman": "people", "crocodile": "reptiles",
               "dinosaur": "reptiles", "lizard": "reptiles", "snake": "reptiles", "turtle": "reptiles",
               "hamster": "small_mammals", "mouse": "small_mammals", "rabbit": "small_mammals",
               "shrew": "small_mammals", "squirrel": "small_mammals", "maple_tree": "trees", "oak_tree": "trees",
               "palm_tree": "trees", "pine_tree": "trees", "willow_tree": "trees", "bicycle": "vehicles_1", "bus": "vehicles_1",
               "motorcycle": "vehicles_1", "pickup_truck": "vehicles_1", "train": "vehicles_1",
               "lawn_mower": "vehicles_2", "rocket": "vehicles_2", "streetcar": "vehicles_2", "tank": "vehicles_2",
               "tractor": "vehicles_2"}
    parent_list_dict = {class_name_to_class_id[key]: superclass_name_to_superclass_id[val] for key, val in class_name_to_superclass_name.items()}
    parent_list = [-1 for _ in range(len(parent_list_dict))]
    for key, val in parent_list_dict.items():
        parent_list[key] = val
    return parent_list

def customSampler(label_imgidx_dict, portion_to_keep):
    final_indices = []
    for label in range(len(portion_to_keep)):
        num_to_keep = int(len(label_imgidx_dict[label]) * portion_to_keep[label])
        indices = random.sample(label_imgidx_dict[label], num_to_keep)
        final_indices.append(indices)
    flat_list = [item for sublist in final_indices for item in sublist]
    return flat_list