import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import ttest_ind
from datasets import load_dataset

from kn_code.patch import unpatch_ff_layers
from kn_code import load_model
from time import strftime
import argparse
import ast
import os
import pandas as pd
from tqdm import tqdm


save_dir = "/scratch/"
os.makedirs(save_dir, exist_ok=True)
PROMPT_DIR = 'prompt_instructions'


def know_bias_analysis_temp_all_new(model_name, condition, num_prob):
    kn = load_model(model_name)
    # path = 'show_bias_res/' + condition
    # data_dir = os.path.join(path)

    DATA_DIR = 'data'

    # th_val = 0.1
    # p_val = 0.49

    th_val = args.threshold
    p_val = args.p_value

    if "llama" in args.model_name:
        model_name = args.model_name[11:]
    elif "Qwen" in args.model_name:
        model_name = args.model_name[5:]
    elif "mistral" in args.model_name:
        model_name = args.model_name[10:]
    elif 'gemma' in args.model_name:
        model_name = args.model_name.split('/')[1]
    else:
        model_name = args.model_name

    if args.add == 'all_extend':
        _df = pd.read_csv(os.path.join(DATA_DIR, 'know_bias_data_all_combine_extend.csv'))
    elif args.add == 'extend':
        _df = pd.read_csv(os.path.join(DATA_DIR, 'know_bias_data_extend.csv'))
    

    att_scores = []
    att_scores_tmp = []
    group_dic = set()

    start, end = args.start, args.end

    for idx, row in tqdm(_df.iterrows(), total=len(_df)):
        question = row['question'] # think question

        target = row['gold']

        q_idx = row['q_index']
        q_polarity = row['q_polarity']

        with(open(f'{PROMPT_DIR}/evaluate_know_new.txt', 'r')) as f:
                instruction = f.read()

        q_add = f"{question}"

        source_prompt = f"""{instruction}
Question: {q_add}"""

        if (q_idx, q_polarity) not in group_dic:
            if group_dic:
                print('=== GROUP WISE START ===')
                group_neurons_ = kn.get_kns_func(att_scores_tmp, threshold=th_val, p=p_val)
                print('=== GROUP WISE END ===')

                group_dic.add((q_idx, q_polarity))
                print(f'Processing group: {q_idx}, {q_polarity}')

                att_scores_tmp = []
            else:
                group_dic.add((q_idx, q_polarity))
                print(f'First Processing group: {q_idx}, {q_polarity}')

                att_scores_tmp = []

        att_score = kn.search_kns_modi(source_prompt, target, n_prompts=num_prob, batch_size=1, verbose=args.verbose) # batch size 1
        att_scores.append(att_score)
        att_scores_tmp.append(att_score)

        raw_neurons = kn.get_raw_n([att_score], threshold=th_val)

    ### for last one
    print('=== GROUP WISE START ===')
    group_neurons = kn.get_kns_func(att_scores_tmp, threshold=th_val, p=p_val)
    print('=== GROUP WISE END ===')
    
    print('=== OVERAL EVAL ===')
    all_combined_neurons = kn.get_kns_func(att_scores, threshold=th_val, p=p_val)

    save_path = os.path.join(save_dir, 'expected_ans_' + str(args.ans) + '_' + str(th_val) + '_' + str(p_val) + '_' + model_name + '_att_scores_know_bias_new_no_ct' + args.add + '.npy')

    np.save(save_path, torch.stack(att_scores).cpu().numpy())  #



def main():
    # python evaluate.py --model_name=meta-llama/Meta-Llama-3.1-8B-Instruct --dataset=bbq --num_probs=10
    parser = argparse.ArgumentParser()
    # all_cat = ["Age", "Disability_status", "Gender_identity", "Nationality", "Physical_appearance", "Race_ethnicity", "Race_x_SES", "Race_x_gender", "Religion", "Sexual_orientation"]
    parser.add_argument('--condition', type=str, default='Gender_identity') # bias cat
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--num_probs', '-n', type=int, default=5)
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--T', type=str, default="")
    parser.add_argument('--ans', type=int, default=0) # 0 is expected ans, 1 is ans1, 2 is ans2
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('--kos', type=str, default='show') ###
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--p_value', type=float, default=0.49)
    parser.add_argument('--add', type=str, default='')
    # parser.add_argument('--bias_cat', type=str, default='Age') #
    
    global args
    args = parser.parse_args()
    # t = strftime('%Y%m%d-%H%M')

    args.T = strftime('%Y%m%d-%H%M')
        
    print(args)

    know_bias_analysis_temp_all_new(args.model_name, args.condition, args.num_probs) # new, only ans0, ans1, yes or no ans


if __name__ == '__main__':
    main()