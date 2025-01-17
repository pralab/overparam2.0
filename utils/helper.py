import matplotlib.pyplot as plt
import numpy as np
from secml.data.loader import CDataLoaderMNIST
from scipy.special import softmax
from secml.ml.peval.metrics import CMetricAccuracy
from secml.array import CArray
from secml.data import CDataset
from secml.ml import CClassifier

from utils.cifar_sampler import cifar_sampler
from tinynet import ExpandableCNN_mnist, ExpandableFcReLu_mnist
from utils.folder import PLOT_FOLDER


def performance_score_model(classifier: CClassifier, tr_dataset: CDataset, ts_dataset: CDataset):
    tr_y_pred = classifier.predict(tr_dataset.X)
    ts_y_pred = classifier.predict(ts_dataset.X)
    print("ORIGINAL y in performance_score_model: ", ts_dataset.Y)
    print("PREDICTED y in performance_score_model: ", ts_y_pred)
    tr_acc = CMetricAccuracy().performance_score(tr_dataset.Y, tr_y_pred)
    ts_acc = CMetricAccuracy().performance_score(ts_dataset.Y, ts_y_pred)
    tr_error = 1 - tr_acc
    ts_error = 1 - ts_acc
    return tr_acc, ts_acc, tr_error, ts_error


def log_mse_loss(clf: CClassifier, data: CDataset):
    scores = clf.decision_function(data.X)
    predictions = softmax(scores.tondarray(), axis=1)
    labels = data.Y.tondarray()
    y = np.zeros((labels.size, labels.max() + 1))
    y[np.arange(labels.size), labels] = 1
    loss = np.log(np.mean([(f - yi)**2 for f, yi in zip(predictions, y)]))
    print(f"Loss: {loss}")
    return loss

# def test_models(classifiers: list[CClassifier], tr_dataset: CDataset, ts_dataset: CDataset):
def test_models(classifiers, tr_dataset, ts_dataset):
    ts_errors = []
    tr_errors = []
    tr_accuracy = []
    ts_accuracy = []
    # clf_perf = {}
    for c in classifiers:
        tr_acc, ts_acc, tr_err, ts_err = performance_score_model(c, tr_dataset, ts_dataset)
        # param = sum([i.numel() for i in list(c.model.parameters())])
        # clf_perf[param]['tr_acc'] = tr_acc
        # clf_perf[param]['ts_acc'] = ts_acc
        tr_accuracy.append(tr_acc)
        ts_accuracy.append(ts_acc)
        ts_errors.append(ts_err)
        tr_errors.append(tr_err)
    tr_accuracy = CArray(tr_accuracy)
    ts_accuracy = CArray(ts_accuracy)
    ts_errors = CArray(ts_errors)
    tr_errors = CArray(tr_errors)

    return tr_errors, ts_errors, tr_accuracy, ts_accuracy


def plot_performance(classifiers,
                             tr_dataset: CDataset,
                             ts_dataset: CDataset,
                             x_axis: CArray,
                             xlabel: str = "",
                             title: str = "",
                             savefig: str = None):
    
    print("classifier: ", type(classifiers))
    print("tr_dataset: ", type(tr_dataset))
    print("ts_dataset: ", type(ts_dataset))
    tr_error, ts_error, tr_accuracy, ts_accuracy = test_models(classifiers, tr_dataset, ts_dataset)
    plt.plot(x_axis.tondarray(), tr_error.tondarray(), label="tr error", c='r')
    plt.plot(x_axis.tondarray(), ts_error.tondarray(), label="ts error", c='b')
    plt.xlabel(xlabel)
    plt.ylabel("Errors")   
    plt.title(title)
    plt.legend()
    if savefig:
        plt.savefig(str(PLOT_FOLDER / f"{savefig}.jpg"))
    plt.show()

    return tr_accuracy, ts_accuracy

