from secml.data.loader import CDataLoaderMNIST, CDataLoaderCIFAR10
import pandas as pd
import pickle
import random
import numpy as np
from secml.data import CDataset
from utils.folder import SUBSETCIFAR

SEED = 123

cifar = CDataLoaderCIFAR10()._load(train_files=['data_batch_1'], test_files=['test_batch'], meta_file='batches.meta', labels_key=b'labels', class_names_key=b'label_names')


train = cifar[0]
test = cifar[1]

label = list(set(train.Y))
n_labels = len(label)
num_samples = 3000

def get_num_samples_per_class(num_samples, n_labels):
    n_samples_per_class = num_samples // n_labels
    return n_samples_per_class


def get_labelwise_index_list(ds_part):
    
    idx_lbl_0, idx_lbl_1, idx_lbl_2, idx_lbl_3, idx_lbl_4, idx_lbl_5, idx_lbl_6, idx_lbl_7, idx_lbl_8, idx_lbl_9 = [], [], [], [], [], [], [], [], [], []
    print("in func get_labelwise_index_list ")
    if ds_part == "train":
        print("Count of labels in TRAIN dataset")
        print(pd.DataFrame(train.Y).value_counts())
        [idx_lbl_0.append(idx) for idx, i in enumerate(train.Y) if i == 0]
        [idx_lbl_1.append(idx) for idx, i in enumerate(train.Y) if i == 1]
        [idx_lbl_2.append(idx) for idx, i in enumerate(train.Y) if i == 2]
        [idx_lbl_3.append(idx) for idx, i in enumerate(train.Y) if i == 3]
        [idx_lbl_4.append(idx) for idx, i in enumerate(train.Y) if i == 4]
        [idx_lbl_5.append(idx) for idx, i in enumerate(train.Y) if i == 5]
        [idx_lbl_6.append(idx) for idx, i in enumerate(train.Y) if i == 6]
        [idx_lbl_7.append(idx) for idx, i in enumerate(train.Y) if i == 7]
        [idx_lbl_8.append(idx) for idx, i in enumerate(train.Y) if i == 8]
        [idx_lbl_9.append(idx) for idx, i in enumerate(train.Y) if i == 9]

    elif ds_part == "test":
        print("Count of labels in TEST dataset")
        print(pd.DataFrame(test.Y).value_counts())
        [idx_lbl_0.append(idx) for idx, i in enumerate(test.Y) if i == 0]
        [idx_lbl_1.append(idx) for idx, i in enumerate(test.Y) if i == 1]
        [idx_lbl_2.append(idx) for idx, i in enumerate(test.Y) if i == 2]
        [idx_lbl_3.append(idx) for idx, i in enumerate(test.Y) if i == 3]
        [idx_lbl_4.append(idx) for idx, i in enumerate(test.Y) if i == 4]
        [idx_lbl_5.append(idx) for idx, i in enumerate(test.Y) if i == 5]
        [idx_lbl_6.append(idx) for idx, i in enumerate(test.Y) if i == 6]
        [idx_lbl_7.append(idx) for idx, i in enumerate(test.Y) if i == 7]
        [idx_lbl_8.append(idx) for idx, i in enumerate(test.Y) if i == 8]
        [idx_lbl_9.append(idx) for idx, i in enumerate(test.Y) if i == 9]


    idx_list = [idx_lbl_0, idx_lbl_1, idx_lbl_2, idx_lbl_3, idx_lbl_4, idx_lbl_5, idx_lbl_6, idx_lbl_7, idx_lbl_8, idx_lbl_9]

    return idx_list



def get_required_indices(num_samples, n_labels, ds_part):
    idx_list = get_labelwise_index_list(ds_part)
    n_samples_per_class = get_num_samples_per_class(num_samples, n_labels)
    print("Num of samples per class: ", n_samples_per_class)
    
    required_idx = []
    for i in range(len(idx_list)):
        required_idx.extend(idx_list[i][:n_samples_per_class])
    
    return required_idx


def cifar_sampler(train_size, test_size):
    
    random.seed(SEED)
    tr_reqd_idx = get_required_indices(num_samples=train_size, n_labels=n_labels, ds_part="train")
    ts_reqd_idx = get_required_indices(num_samples=test_size, n_labels=n_labels, ds_part="test")

    random.shuffle(tr_reqd_idx)
    random.shuffle(ts_reqd_idx)

    x_train = train.X[tr_reqd_idx, :]
    y_train = train.Y[tr_reqd_idx]

    x_test = test.X[ts_reqd_idx, :]
    y_test = test.Y[ts_reqd_idx]

    train_subset = CDataset(x_train, y_train)
    test_subset = CDataset(x_test, y_test)

    return(train_subset, test_subset)
