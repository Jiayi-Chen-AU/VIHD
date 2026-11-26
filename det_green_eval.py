import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
from green_score import GREEN
import pdb
import argparse

os.environ["HF_HOME"] = "/home/jchen/dv94_scratch2/jychen/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/home/jchen/dv94_scratch2/jychen/huggingface"
os.environ["HUGGINGFACE_HOME"] = "/home/jchen/dv94_scratch2/jychen/huggingface"

def eval_t2t_csv(in_csv_file, out_csv_file):
    model_name = "StanfordAIMI/GREEN-radllama2-7b"
    green_scorer = GREEN(model_name, output_dir=".")
    in_df = pd.read_csv(in_csv_file, usecols=[0, 1, 2, 3]).astype(str)   # img_idx, question, ref_answer, gen_answer

    in_df['ref_answer'] = in_df['question'] + " " + in_df['ref_answer']
    in_df['gen_answer'] = in_df['question'] + " " + in_df['gen_answer']
    refs_list = in_df['ref_answer'].tolist()
    hyps_list = in_df['gen_answer'].tolist()

    mean, std, green_score_list, summary, result_df = green_scorer(refs_list, hyps_list)

    result_df = result_df.drop(['reference', 'predictions'], axis=1)
    in_df.reset_index(drop=True, inplace=True)
    result_df.reset_index(drop=True, inplace=True)
    out_df = pd.concat([in_df, result_df], axis=1)

    out_df.to_csv(out_csv_file, index=False)

# conda activate env_green (!!! Set up a dedicated virtual environment for the Green model to prevent conflicts with environments used for other MLLMs.)
# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 green_eval.py
# deactivate



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="slake")
    parser.add_argument("--model", type=str, default="lingshu")
    parser.add_argument("--type", type=str, default="all")

    return parser.parse_args()


# python main_hall_det.py
if __name__ == '__main__':
    args = get_args()
    in_csv_file = f"det_outputs_test/{args.dataset}_{args.model}_vihd_{args.type}.csv"
    out_csv_file = f"det_green_test/{args.dataset}_{args.model}_green_{args.type}.csv"

    in_dir_name = os.path.dirname(in_csv_file)
    if in_dir_name:
        os.makedirs(in_dir_name, exist_ok=True)

    out_dir_name = os.path.dirname(out_csv_file)
    if out_dir_name:
        os.makedirs(out_dir_name, exist_ok=True)

    eval_t2t_csv(in_csv_file, out_csv_file)