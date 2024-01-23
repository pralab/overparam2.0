import torch
from torch import nn
import numpy as np
import argparse
import time
from autoattack import AutoAttack
from secml.adv.attacks import CFoolboxPGDL2, CAttackEvasionPGDLS
from secml.adv.seceval import CSecEval, CSecEvalData
from secml.array import CArray
from data_utils import load_pytorch_dataset, pytorch_ds_to_secml_ds
from model_utils import get_models_and_path, pretrain_secml, get_pgd_attack_hyperparams
from plot_utils import plot_base_performance, plot_robustness_performance
import config
from folder import PGDL2_CNN_MNIST, AA_CNN_MNIST, PGDL2_FCRELU_MNIST, AA_FCRELU_MNIST

"""
Sample command python run.py --ds mnist --model cnn --attack pgdl2 --train 20
"""

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

def runPGDL2attack_secml(attack, parameters, clfs, tr_dataset, ts_dataset, folder_pretrained_model, epsilons, solver_params, dist, lb, ub, y_target):
    for p, clf in zip(parameters, clfs):

        if not (folder_pretrained_model/f"security_evaluation_{p}.gz").exists():

            pgd_l2_attack = CAttackEvasionPGDLS(classifier=clf, 
                                                double_init_ds=tr_dataset,
                                                double_init=True,
                                                y_target=y_target, 
                                                distance=dist,
                                                dmax=0,
                                                lb = lb, ub = ub,
                                                solver_params=solver_params)
            pgd_l2_attack.verbose = 0
            sec_eval_pgd = CSecEval(pgd_l2_attack, "dmax", epsilons, save_adv_ds=False)
            
            print("[INFO] Attack Starterd ...")
            starting_time = time.time()
            sec_eval_pgd.run_sec_eval(ts_dataset)
            end_time = time.time()
            print(f"Time taken to run security evaluation for {attack} attack: {end_time-starting_time}")
            
            sec_eval_pgd.sec_eval_data.save(folder_pretrained_model/f"security_evaluation_{p}.gz")
        else:
            print(f"Security evaluation for {p} param model already computed. All set for robustness plots")
    

def runAutoAttack_pytorch(attack, parameters, clfs, sec_eval_folder, epsilons, test_loader):
    
    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0)
    
    for p, clf in zip(parameters, clfs):
        if not (sec_eval_folder/f"security_evaluation_{p}.gz").exists():
            csecevaldata_obj =  CSecEvalData()
            csecevaldata_obj.param_name = "epsilons"
            csecevaldata_obj.param_values = epsilons
            csecevaldata_obj.Y = CArray(y_test.numpy()).deepcopy()  # original labels
            csecevaldata_obj.Y_pred =  []
            csecevaldata_obj.scores = []
            csecevaldata_obj.time = []
            nnet = clf.model

            att_pred = []
            cum_att_scores = []
            for eps in epsilons:
                AA = AutoAttack(nnet, norm="L2", eps=eps, seed=SEED, version="standard", device=device)
                print(f"\nStart autoattack for epsilon: {eps}")
                start_time = time.time()
                x_adv, y_adv = AA.run_standard_evaluation(x_test, y_test, batch_size, return_labels=True)
                # att_scores = nnet(x_adv)
                # cum_att_scores.append(att_scores)
                # pred_proba = nn.Softmax(dim=1)(att_scores)
                # att_pred.append(CArray(pred_proba.argmax(1)))  
                att_pred.append(CArray(y_adv))
                end_time = time.time()
                
            
            csecevaldata_obj.scores.append(cum_att_scores)
            csecevaldata_obj.Y_pred.append(att_pred)
            csecevaldata_obj.time.append(end_time- start_time)

            csecevaldata_obj.save(sec_eval_folder/f"security_evaluation_{p}.gz")
        else:
            print(f"Security evaluation for {p} param model already computed. All set for robustness plots")


if __name__ == "__main__":

    SEED = 121

    parser = argparse.ArgumentParser(description="Run multiple trainings with growing network capacity")
    parser.add_argument("--ds", type=str, help="select dataset : mnist or cifar10",
        choices={"mnist", "cifar10"}, dest="dataset_name", default="mnist")
    parser.add_argument(
        "--model", type=str, help="select model to train: cnn or fcrelu", choices={"cnn", "fcrelu"}, dest="model", default="cnn")
    parser.add_argument(
        "--epochs", type=int, help="The number of epochs to train, if not given, the value in the config file is taken", dest="epochs")
    parser.add_argument(
        "--attack", type=str, help="choose the type of attack: pgdl2, autoattck", choices={"pgdl2", "autoattack"}, dest="attack_name", default="pgdl2")
    parser.add_argument(
        "--train", type=int, help="give the size o of the training set", dest="train_subset", default=3000)

    
    args = parser.parse_args()
    print(f"seed: {SEED}")

    ds_name, model_name, attack, train_subset = args.dataset_name, args.model, args.attack_name, args.train_subset
    epoch, batch_size, lr, steps, stepsize = config.EPOCH1, config.BATCHSIZE3, config.LR1, config.STEPS, config.STEPSIZE
    # batch_size needs to me a number that is a factor of train and test subset number, BATCHSIZE3 = 20
    
    epochs = args.epochs if args.epochs else config.EPOCH1
    if ds_name == "mnist":
        input_shape = config.MNIST_INPUT_SHAPE
        test_subset = config.MNIST_TESTSIZE
        solver_params, noise_type, lb, ub, y_target = get_pgd_attack_hyperparams(ds_name)
        if model_name == "cnn":
            expansions = [1, 2, 4, 6, 8, 10, 15, 20, 25, 30]
            if attack == "pgdl2":
                epsilons = np.linspace(0.3, 3.0, 5)
                sec_eval_folder = PGDL2_CNN_MNIST
            elif attack == "autoattack":
                epsilons = np.linspace(0.05, 2.0, 5)
                sec_eval_folder = AA_CNN_MNIST
        elif model_name == "fcrelu":
            epsilons = np.linspace(0.1, 1.5, 5)
            expansions = [4, 6, 8, 10, 15, 20, 25, 30, 35, 40]
            if attack == "pgdl2":
                sec_eval_folder = PGDL2_FCRELU_MNIST
            elif attack == "autoattack":
                sec_eval_folder = AA_FCRELU_MNIST
    elif ds_name == "cifar10":
        input_shape = config.CIFAR10_INPUT_SHAPE
        test_subset = config.CIFAR10_TESTSIZE
        solver_params, noise_type, lb, ub, y_target = get_pgd_attack_hyperparams()
    
    print(f"[INFO] Loading {ds_name} in pytorch...")
    dataset = load_pytorch_dataset(SEED, ds_name, train_subset, test_subset, batch_size, ds_normalization=False)
    print(f"[INFO] loding {model_name} for given model name...")
    model_folder, clf_names, model = get_models_and_path(ds_name, model_name, expansions)
    print("[INFO] Converting pytorch dataset to secml format...")
    tr_secml = pytorch_ds_to_secml_ds(dataset.train_loader, batch_size)
    ts_secml = pytorch_ds_to_secml_ds(dataset.test_loader, batch_size)
    print("[INFO] Pretraining/Loading existing network ...")
    clfs = pretrain_secml(device, model_folder, clf_names, expansions, tr_secml, model, input_shape, config.OUTCLASSES, epoch, batch_size, lr)
    print(f"[INFO] Plotting base performance for {ds_name}-{model}...")
    tr_acc, ts_acc = plot_base_performance(clfs, tr_secml, ts_secml, ds_name, model_name)
    
    parameters = [sum([i.numel() for i in list(clf.model.parameters())]) for clf in clfs]
    solver_params, noise_type, lb, ub, y_target = get_pgd_attack_hyperparams(ds_name)

    print("[INFO] Attacking the model begins ...")
    if attack == "pgdl2":
        runPGDL2attack_secml(attack, parameters, clfs, tr_secml, ts_secml, sec_eval_folder, epsilons, solver_params, noise_type, lb, ub, y_target)
        print(f"\n[INFO] Plotting security evaluation plots {ds_name}-{model} - {attack}...")
        plot_robustness_performance(ds_name, model_name, attack, parameters, sec_eval_folder, ts_acc)
    elif attack == "autoattack":
        runAutoAttack_pytorch(attack, parameters, clfs, sec_eval_folder, epsilons, dataset.test_loader)
        print(f"\n[INFO] Plotting security evaluation plots {ds_name}-{model} - {attack}...")
        plot_robustness_performance(ds_name, model_name, attack, parameters, sec_eval_folder, ts_acc)

    
    
    



## TO DO
## Add test_subset size  -- DONE
## check the functiont hat converts pytorch datasetr to secml   -- DONE
## in pgdl2 attack, try with CAttackEvasionPGDLS insteaf of CFoolboxPGDL2  -- DONE
## in autoattack, the expected inbut is eith 3D or 4D, curretly its in the shape of (batch_size, img_size): [20, 784] - check the loader function and load in orig pytorch format without reshapong and check whether the func pytorch_ds_to_secml is able to correctly change the datashape else use reshape..