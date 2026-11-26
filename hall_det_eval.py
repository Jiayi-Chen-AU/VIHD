import torch
import numpy as np
import sklearn.metrics as metrics
import pandas as pd
from sklearn.metrics import auc
import pdb
import argparse
import csv

def compute_aug(hall_scores, hall_labels, resolution=100):
    """
    Compute the AUG.

    Parameters:
        hall_scores (list or np.ndarray):
        hall_labels (list or np.ndarray): 
        resolution (int): Number of cut-off percentages to evaluate (default is 100).

    Returns:
        float: The computed AUG value.
    """
    # Ensure inputs are numpy arrays
    hall_scores = np.array(hall_scores)
    hall_labels = np.array(hall_labels)

    # Sort samples by hallucination scores in ascending order
    sorted_indices = np.argsort(hall_scores)
    sorted_labels = hall_labels[sorted_indices]

    # Initialize variables for AURAC computation
    total_samples = len(hall_scores)
    cut_offs = np.linspace(0, 1, resolution + 1)  # Cut-off percentages (0% to 100%)
    green_list = []

    for cut_off in cut_offs:
        # Number of samples in the confident subset
        num_selected = int(cut_off * total_samples)
        
        # Compute accuracy for the confident subset
        if num_selected > 0:
            confident_subset = sorted_labels[:num_selected]
            green_mean = np.mean(confident_subset)
        else:
            green_mean = 0
        green_list.append(green_mean)

    aug = auc(cut_offs, green_list)
    return aug


def get_threshold_f1(label, score):
    thresholds = np.unique(score)
    best_f1 = -1.0
    best_t = None

    for t in thresholds:
        f1 = metrics.f1_score(label, score>=t)

        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return best_t, best_f1


def get_auc_aug(csv_hall_label, column_hall_label, csv_hall_score, column_hall_score, val_hall_label, val_hall_score, uncertainty_flag):
    # hallucination label and score for the test set
    hall_label = pd.read_csv(csv_hall_label, usecols=[column_hall_label])
    hall_label = torch.tensor(hall_label[column_hall_label].values)
    hall_label_int = (hall_label < 1.0).long() # green<1: hallucination, green==1: non-hallucination

    hall_score = pd.read_csv(csv_hall_score, usecols=[column_hall_score])
    hall_score = torch.tensor(hall_score[column_hall_score].values)
    hall_score_norm = (hall_score - hall_score.min()) / (hall_score.max() - hall_score.min() + 1e-8)
    if uncertainty_flag == False:
        hall_score_norm = 1.0 - hall_score_norm

    # AUC
    auc_score = metrics.roc_auc_score(hall_label_int, hall_score_norm)

    # AUG
    aug_score = compute_aug(hall_score_norm.tolist(), hall_label.tolist())



    # open-ended questions
    q_type = pd.read_csv(csv_hall_score, usecols=["answer_type"])
    open_mask = (q_type["answer_type"] == "open").values
    
    hall_label = pd.read_csv(csv_hall_label, usecols=[column_hall_label])
    hall_label = torch.tensor(hall_label[column_hall_label].values[open_mask])
    hall_label_int = (hall_label < 1.0).long() # green<1: hallucination, green==1: non-hallucination

    hall_score = pd.read_csv(csv_hall_score, usecols=[column_hall_score])
    hall_score = torch.tensor(hall_score[column_hall_score].values[open_mask])
    hall_score_norm = (hall_score - hall_score.min()) / (hall_score.max() - hall_score.min() + 1e-8)
    if uncertainty_flag == False:
        hall_score_norm = 1.0 - hall_score_norm

    # AUC
    open_auc_score = metrics.roc_auc_score(hall_label_int, hall_score_norm)

    # AUG
    open_aug_score = compute_aug(hall_score_norm.tolist(), hall_label.tolist())
    
    return round(auc_score * 100, 2), round(aug_score * 100, 2), round(open_auc_score * 100, 2), round(open_aug_score * 100, 2)


def main(
    dataset_name="radvqa",
    model_name="medgemma",
    qtype="open",
    column_hall_label="green_score",
    column_hall_score="SE", # SE, VASE, RadFlag
    uncertainty_flag=True, # True when hall_score indicates uncertainty (SE, VASE), False when hall_score indicates confidence (RadFlag).
):
    csv_hall_label=f"det_green_test/{dataset_name}_{model_name}_green_{qtype}.csv"
    csv_hall_score=f"det_outputs_test/{dataset_name}_{model_name}_vihd_{qtype}.csv"
    val_hall_label=f"det_green_val/{dataset_name}_{model_name}_green_{qtype}.csv",
    val_hall_score=f"det_outputs_val/{dataset_name}_{model_name}_vihd_{qtype}.csv",


    auc_score, aug_score, open_auc_score, open_aug_score = get_auc_aug(csv_hall_label, column_hall_label, 
                           csv_hall_score, column_hall_score, 
                           val_hall_label, val_hall_score,
                           uncertainty_flag)
    print(f'Method:\t{column_hall_score}\topen_AUC: {open_auc_score}%\topen_AUG: {open_aug_score}%\tAUC: {auc_score}%\tAUG: {aug_score}%\t')
    return open_auc_score, open_aug_score, auc_score, aug_score


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="vqa-rad")
    parser.add_argument("--model", type=str, default="medgemma")
    parser.add_argument("--type", type=str, default="open")

    return parser.parse_args()

# python hall_det_eval.py
if __name__ == "__main__":
    args = get_args()
    dataset_name = args.dataset
    model_name = args.model
    qtype= args.type
    method_list = ["VIHD"]
    flag_list = [True]
    auc_list = []
    aug_list = []
    open_auc_list = []
    open_aug_list = []
    for i in range(len(method_list)):
        method = method_list[i]
        flag = flag_list[i]
        open_auc_score, open_aug_score, auc_score, aug_score = main(dataset_name=dataset_name, model_name=model_name, qtype=qtype, column_hall_score=method, uncertainty_flag=flag)
        auc_list.append(auc_score)
        aug_list.append(aug_score)
        open_auc_list.append(open_auc_score)
        open_aug_list.append(open_aug_score)


    csv_file = f"det_result/{dataset_name}_{model_name}.csv"
    dir_name = os.path.dirname(csv_file)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Flag", "Open AUC", "Open AUG", "AUC", "AUG"])
        for i in range(len(method_list)):
            writer.writerow([
                method_list[i],
                flag_list[i],
                open_auc_list[i],
                open_aug_list[i],
                auc_list[i],
                aug_list[i]
            ])

    print(f"Results saved to {csv_file}")
        
    
    
