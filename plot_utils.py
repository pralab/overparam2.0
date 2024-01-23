
from secml.ml.peval.metrics import CMetricAccuracy
from secml.figure import CFigure
from secml.array import CArray
from secml.adv.seceval import CSecEval, CSecEvalData
from helper import plot_performance
from folder import PLOT_FOLDER

def plot_base_performance(clfs, tr_dataset, ts_dataset, ds, network):
    parameters = [sum([i.numel() for i in list(clf.model.parameters())]) for clf in clfs]
    tr_acc, ts_acc = plot_performance(clfs, tr_dataset, ts_dataset, CArray(parameters), "Network parameters",
                                title=f"Error rate of {network} on {ds} by varying numbers of parameters",
                                savefig= str(PLOT_FOLDER / f"{ds}_{network}_base_performance_trainsize{tr_dataset.X.shape[0]}_test{ts_dataset.X.shape[0]}"))
    print(f"BASE PERFORMANCE\nTrain accuracy: {tr_acc},\t Testing accuracy: {ts_acc}")
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
            print("in mnist")
            if attack == "autoattack":
                print("in mnist AA")
                att_pred = sec_eval_data.Y_pred[0]  
            else:
                att_pred = sec_eval_data.Y_pred
        full_range_acc = base_ts_acc[i].tolist()

        print("epsilons: ", sec_eval_data.param_values)
        # print("y_true: ", y_true)
        # print("att_pred: ", att_pred)
        # print("first vector of att_pred: ", att_pred[0])
        # print("type att_pred[0]: ", type(att_pred[0]))
        # print("epsilons.size: ", epsilons.size)
        
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
