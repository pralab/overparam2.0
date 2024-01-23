from __future__ import print_function

import os
import time
import datetime
from collections import namedtuple
from secml.array import CArray
from secml.data import CDataset
import random
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, sampler
import torch

# fixed no augmentation on training dataset
# fixed cifar normalization values

Dataset = namedtuple("Dataset", ["train_loader", "val_loader", "test_loader"])

def load_pytorch_dataset(seed, dataset, train_subset_size, test_subset_size, batch_size,
                ds_normalization=True):

    if dataset.lower() == "mnist":
        ds = datasets.MNIST
    elif dataset == "cifar10":
        ds = datasets.CIFAR10
    else:
        raise ValueErrsheduleror("dataset unknown")

    train_transform = get_train_transform(dataset, ds_normalization)

    train_set = ds(root='./data',
                   download=True,
                   train=True,
                   transform=train_transform)
    val_subset_size = int(0.2 * train_subset_size)
    random.seed(seed)
    random_train_indices =random.sample(range(len(train_set)),train_subset_size)
    print("num tr samples: ", len(random_train_indices))
    random_val_indices = np.random.choice(len(train_set),
                                          size=val_subset_size)
    print("num val samples: ", len(random_val_indices))
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              num_workers=0,
                              sampler=sampler.SubsetRandomSampler(random_train_indices))
    val_loader = DataLoader(train_set,
                            batch_size=batch_size,
                            num_workers=0,
                            sampler=sampler.SubsetRandomSampler(random_val_indices))

    test_transform = get_test_transform(dataset, ds_normalization)
    test_set = ds(root='./data',
                  download=True,
                  train=False,
                  transform=test_transform)
    
    random_test_indices = random.sample(range(len(test_set)),test_subset_size)
    print("num ts samples: ", len(random_test_indices))
    
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=0,
                             sampler=sampler.SubsetRandomSampler(random_test_indices))
    
    dataset = Dataset(train_loader=train_loader,
                      val_loader=val_loader, test_loader=test_loader)

    return dataset

def get_train_transform(dataset, ds_normalization):

    if ds_normalization is True:
        if dataset == "mnist":
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(
                                                (0.1307,), (0.3081,)),
                                            transforms.Lambda(lambda img: img.reshape(-1))
                                            ])
        else:
            transform = transforms.Compose([transforms.RandomCrop(28, padding=0),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4916, 0.4824, 0.4467),
                                                                 (0.0299, 0.0295, 0.0318))
                                            ])
        return transform


    else:
        if dataset == "mnist":
            transform = transforms.Compose([transforms.ToTensor()]) #,transforms.Lambda(lambda img: img.reshape(-1))])
        else:
            transform = transforms.Compose([transforms.RandomCrop(28, padding=0),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            ])
        return transform

def get_test_transform(dataset, ds_normalization):

    """
    Transofrmations for test data for mnist and cifar10
    """

    if ds_normalization is True:
        if dataset == "mnist":
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(
                                                (0.1307,), (0.3081,)),
                                            transforms.Lambda(lambda img: img.reshape(-1))
                                            ])
        else:
            transform = transforms.Compose([
                                            transforms.Resize(28),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4916, 0.4824, 0.4467),
                                                                 (0.0299, 0.0295, 0.0318))
                                            ])
        return transform
    else:
        if dataset == "mnist":
            transform = transforms.Compose([transforms.ToTensor()]) #,transforms.Lambda(lambda img: img.reshape(-1))])
        else:
            transform = transforms.Compose([
                                            transforms.Resize(28),
                                            transforms.ToTensor(),
                                            ])
        return transform
    
def pytorch_ds_to_secml_ds(ds_loader,batch_size):
    """
    Get a pytorch dataset loader and return a CDataset (the secml data
    structure for datasets)

    Input:
    ds_loader: dict 
        dataset.train_loader or dataset.test_loader
    batch_size: int
        batchsize for secml framework
    """
    secml_ds = None
    img_size = None
    for img, y in ds_loader:
        
        if img_size is None:
            # check the image size
            first_img = img[0,:]
            img_size = first_img.reshape(-1)
            img_size = img_size.numpy().size
            # img_size = first_img.numpy().size

        # the pytorch images have 3 dimensions whereas secml work with
        # flatted array.
        secml_img = CArray(img.reshape(batch_size,img_size).numpy())
        secml_y =  CArray(y.numpy())

        current_sample = CDataset(secml_img, secml_y)
        if secml_ds is None:
            secml_ds =  current_sample
        else:
            secml_ds = secml_ds.append(current_sample)

    return secml_ds