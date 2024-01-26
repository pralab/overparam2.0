
import matplotlib.pyplot as plt
from secml.ml.peval.metrics import CMetricAccuracy
from secml.figure import CFigure
from secml.array import CArray
from secml.adv.seceval import CSecEval, CSecEvalData
from secml.array import CArray
from secml.data import CDataset
from secml.ml import CClassifier
from helper import plot_performance
from folder import PLOT_FOLDER

def plot_base_performance(clfs, tr_dataset, ts_dataset, ds, network):
    print("ORIGINAL y in plot_base_performance: ", ts_dataset.Y)
    parameters = [sum([i.numel() for i in list(clf.model.parameters())]) for clf in clfs]
    tr_acc, ts_acc = plot_performance(clfs, tr_dataset, ts_dataset, CArray(parameters), "Network parameters",
                                title=f"Error rate of {network} on {ds} by varying numbers of parameters",
                                savefig= str(PLOT_FOLDER / f"{ds}_{network}_base_performance_trainsize{tr_dataset.X.shape[0]}_test{ts_dataset.X.shape[0]}"))
    print(f"BASE PERFORMANCE\nTrain accuracy: {tr_acc},\t Testing accuracy: {ts_acc}")
    return tr_acc, ts_acc

def plot_base_performance_ambrastyle(clfs, tr_dataset, ts_dataset, ds, network, fig_title, xscale_base):
    parameters = [sum([i.numel() for i in list(clf.model.parameters())]) for clf in clfs]
    tr_err, ts_err, tr_acc, ts_acc = test_models(clfs, tr_dataset, ts_dataset)
    fig = CFigure(3, 5)
    tr_std = tr_acc.std(axis=0).ravel()
    print("tr std: ", tr_std)
    low_tr_acc = tr_err - tr_std
    high_tr_acc = tr_err + tr_std

    ts_std = ts_acc.std(axis=0).ravel()
    print("ts std: ", ts_std)
    low_ts_acc = ts_err - ts_std
    high_ts_acc = ts_err + ts_std

    reduced_tr_error = tr_err
    print("reduced tr error ", reduced_tr_error)
    fig.sp.plot(parameters, reduced_tr_error,
             "-", label="train", marker='o')
    
    reduced_ts_error = ts_err
    fig.sp.plot(parameters, reduced_ts_error,
             "-", label="test", marker='o')
    
    # plot standard deviation
    fig.sp.fill_between(parameters, low_tr_acc,
                        high_tr_acc, alpha=0.5)
    fig.sp.fill_between(parameters, low_ts_acc,
                        high_ts_acc, alpha=0.5)

    print("number of param reduced ", parameters)

    fig.sp.title(fig_title)
    fig.sp.ylabel("Error rate")
    fig.sp.xlabel("Number of parameters")

    print(xscale_base, xscale_base)
    # fig.sp.xscale("log", basex=xscale_base)

    fig.subplots_adjust(left=0.15, right=0.98,
                        bottom=0.18, top=0.93, wspace=0.2, hspace=0.2)

    fig.sp.grid(linestyle='--')

    fig.sp.legend(framealpha = 0.5)

    fig.savefig(str(PLOT_FOLDER /f"base_perf_{ds}-{network}.pdf"))
    return tr_acc, ts_acc


def plot_robustness_performance(ds_name, network, attack, parameters, folder_pretrained_model, base_ts_acc):
    fig = CFigure(3, 5, fontsize=8, markersize=3)
    metric = CMetricAccuracy()
    if ds_name == "mnist":
        if attack == "pgdl2":
            fig_title = "Security evaluation curve for MNIST: PGD attack"
        elif attack == "autoattack":
            fig_title = "Security evaluation curve for MNIST: AutoAttack "
    elif ds_name == "cifar10":
        fig_title = "Security evaluation curve for CIFAR10"
    # full_range_acc = []
    for i, p in enumerate(parameters):
        sec_eval_data = CSecEvalData.load(folder_pretrained_model/f"security_evaluation_{p}.gz")
        epsilons = sec_eval_data.param_values
        y_true = sec_eval_data.Y
        if ds_name == "mnist":
            if attack == "autoattack":
                att_pred = sec_eval_data.Y_pred[0]  
            else:
                att_pred = sec_eval_data.Y_pred
        full_range_acc = base_ts_acc[i].tolist()

        print("epsilons: ", sec_eval_data.param_values)
        
        for eps in range(epsilons.size):
            full_range_acc.append(metric.performance_score(y_true=y_true, y_pred=att_pred[eps]))
        print("\naccuracy scores across all epsilons starting 0: ", full_range_acc)

        rob_error = [1 - x for x in full_range_acc]
        # print("Robustness error = ", rob_error)

        all_epsilons = [0] + epsilons.tolist()
        all_epsilons = ["%.2f" % elem for elem in all_epsilons]
        
        print(f"sec eval plot for MNIST-{network} with {p} param...")
        fig.sp.plot(CArray(all_epsilons).astype(str).tolist(), rob_error, "-", label="{:.2e}".format(p), marker = "o")
    
    fig.sp.title(fig_title)
    fig.sp.ylabel("Error Rate")
    fig.sp.xlabel("epsilons")

    fig.subplots_adjust(left=0.125, right=0.98,
                        bottom=0.15, top=0.93, wspace=0.2, hspace=0.2)
    fig.sp.grid(linestyle='--')
    fig.sp.legend(framealpha = 0.5)
    fig.savefig(str(PLOT_FOLDER /f"sec_eval_{ds_name}-{network}-{attack}.pdf"))


def performance_score_model(classifier: CClassifier, tr_dataset: CDataset, ts_dataset: CDataset):
    tr_y_pred = classifier.predict(tr_dataset.X)
    ts_y_pred = classifier.predict(ts_dataset.X)
    tr_acc = CMetricAccuracy().performance_score(tr_dataset.Y, tr_y_pred)
    ts_acc = CMetricAccuracy().performance_score(ts_dataset.Y, ts_y_pred)
    tr_error = 1 - tr_acc
    ts_error = 1 - ts_acc
    return tr_acc, ts_acc, tr_error, ts_error

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