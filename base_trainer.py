import pickle
import time
import numpy as np
import pandas as pd
import torch.optim
from secml.data.loader import CDataLoaderMNIST
from secml.ml import CClassifierPyTorch
from secml.adv.attacks import CFoolboxPGDL2
from tinynet import ExpandableCNN_mnist
from secml.array import CArray
import random

from folder import PLOT_FOLDER
from helper import plot_performance

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
SEED = 121

def pretrain(device, model_folder, clf_names, expansions, tr_dataset, nn_model, input_shape, output_classes, epoch, batch_size, lr):
    clfs = []
    for name, n in zip(clf_names, expansions):
        if not(model_folder / name).exists():
            net = nn_model(expansion = n, out_classes = output_classes)
            print(f"Training network with {sum([i.numel() for i in list(net.parameters())])}")
            net = net.to(device)
            net.train()
            clf = CClassifierPyTorch(net, input_shape=input_shape,
                                    optimizer=torch.optim.Adam(net.parameters(), lr=lr),
                                    loss=torch.nn.CrossEntropyLoss(), epochs=epoch, batch_size=batch_size)
            clf.fit(tr_dataset.X, tr_dataset.Y)
            print(clf)
            clf.model.eval()
            clf.save(str(model_folder / name))
            clfs.append(clf)
        else:
            clf = CClassifierPyTorch.load(str(model_folder / name))
            clf.model.eval()
            print(f"Loading network with {sum([i.numel() for i in list(clf.model.parameters())])} parameters")
            clfs.append(clf)
    return clfs


