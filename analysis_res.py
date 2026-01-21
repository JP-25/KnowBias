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
from prettytable import PrettyTable



save_dir = "/scratch/"
os.makedirs(save_dir, exist_ok=True)
PROMPT_DIR = 'prompt_instructions'
att_score_dir = 'att_scores_save'
# att_score_dir = 'att_scores_new' # for moti analysis
os.makedirs(att_score_dir, exist_ok=True)



def know_bias_analysis_temp_all_new(model_name, condition, num_prob):
    kn = load_model(model_name)

    DATA_DIR = 'data'

    th_val = args.threshold
    p_val = args.p_value

    if args.add == 'all_extend':
        _df = pd.read_csv(os.path.join(DATA_DIR, 'know_bias_data_all_combine_extend.csv'))
    elif args.add == 'extend':
        _df = pd.read_csv(os.path.join(DATA_DIR, 'know_bias_data_extend.csv'))

    att_scores = []
    att_scores_tmp = []
    group_dic = set()
    table = PrettyTable()
    table_detail = PrettyTable()

    table.field_names = ['bias cat', 'bias res', 'q_index', 'q_polarity', 'group neurons']
    table_detail.field_names = ['idx','question', 'context', 'q_index', 'q_polarity', 'bias cat', 'bias res', 'raw neurons']

    for idx, row in tqdm(_df.iterrows(), total=len(_df)):
        question = row['question']
        # context = row['prompt_context']
        # b_res = row['res']
        q_idx = row['q_index']
        q_polarity = row['q_polarity']
        bias_cat = row['bias_cat']

        if args.ans == 0:

            if args.add == 'all':
                if args.model_name == 'meta-llama/Llama-3.2-3B-Instruct':
                    att_scores_all = np.load(os.path.join(att_score_dir, 'expected_ans_0_0.1_0.49_Llama-3.2-3B-Instruct_att_scores_know_bias_new_n_ctall.npy'), allow_pickle=True)
            elif args.add == 'all_extend':
                if args.model_name == 'meta-llama/Llama-3.2-3B-Instruct':
                    att_scores_all = np.load(os.path.join(att_score_dir, 'expected_ans_0_0.1_0.49_Llama-3.2-3B-Instruct_att_scores_know_bias_new_n_ctall_extend.npy'), allow_pickle=True)
            elif args.add == 'extend':
                if args.model_name == 'meta-llama/Llama-3.2-3B-Instruct':
                    att_scores_all = np.load(os.path.join(att_score_dir, 'expected_ans_0_0.1_0.49_Llama-3.2-3B-Instruct_att_scores_know_bias_new_n_ctextend.npy'), allow_pickle=True)

            else:
                if args.model_name == 'meta-llama/Llama-3.2-3B-Instruct':
                    att_scores_all = np.load(os.path.join(att_score_dir, 'expected_ans_0_0.1_0.49_llama3_2_3B_att_scores_know_bias_new.npy'), allow_pickle=True)
        
        att_scores_tensor = torch.from_numpy(att_scores_all)

        if (q_idx, q_polarity) not in group_dic:
            if group_dic:
                print('=== GROUP WISE START ===')
                group_neurons_ = kn.get_kns_func(att_scores_tmp, threshold=th_val, p=p_val)
                print('=== GROUP WISE END ===')
                table.add_row([prev_bias_cat, 'na', prev_q_idx, prev_q_polarity, group_neurons_])

                group_dic.add((q_idx, q_polarity))
                print(f'Processing group: {q_idx}, {q_polarity}')
                prev_q_idx = q_idx
                prev_q_polarity = q_polarity
                prev_bias_cat = bias_cat
                # prev_b_res = b_res

                att_scores_tmp = []
            else:
                group_dic.add((q_idx, q_polarity))
                print(f'First Processing group: {q_idx}, {q_polarity}')
                prev_q_idx = q_idx
                prev_q_polarity = q_polarity
                prev_bias_cat = bias_cat
                # prev_b_res = b_res

                att_scores_tmp = []

        # att_score = kn.search_kns_modi(source_prompt, target, n_prompts=num_prob, batch_size=1, verbose=args.verbose) # batch size 1
        att_score = att_scores_tensor[idx]
        att_scores.append(att_score)
        att_scores_tmp.append(att_score)
        ###
        print('L188 individual neurons: ', idx)
        raw_neurons = kn.get_raw_n([att_score], threshold=th_val)
        table_detail.add_row([idx, question, 'na', q_idx, q_polarity, bias_cat, 'na', raw_neurons])

    ### for last one
    print('=== GROUP WISE START ===')
    group_neurons = kn.get_kns_func(att_scores_tmp, threshold=th_val, p=p_val)
    print('=== GROUP WISE END ===')
    table.add_row([prev_bias_cat, 'na', prev_q_idx, prev_q_polarity, group_neurons])

    print('=== OVERAL EVAL ===')
    all_combined = kn.get_kns_func(att_scores, threshold=th_val, p=p_val)

    concept_dir = os.path.join('neuron_res')
    os.makedirs(concept_dir, exist_ok=True)

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
    
    save_1 = pd.DataFrame(table._rows, columns=table.field_names)
    save_2 = pd.DataFrame(table_detail._rows, columns=table_detail.field_names)

    save_concepts_dir1 = os.path.join(concept_dir, model_name + '_ans_' + str(args.ans) + '_' + str(th_val) + '_' + str(p_val) +'_group_res_know_bias_' + args.add + 'o_no_ct_tt.csv')

    save_1.to_csv(save_concepts_dir1, index = False, header=True)
    save_concepts_dir2 = os.path.join(concept_dir, model_name + '_ans_' + str(args.ans) + '_' + str(th_val) + '_' + str(p_val) + '_detail_res_know_bias_' + args.add + 'o_no_ct_tt.csv') #
    save_2.to_csv(save_concepts_dir2, index = False, header=True)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', type=str, default='Gender_identity') # bias cat
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--num_probs', '-n', type=int, default=5)
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--T', type=str, default="")
    parser.add_argument('--ans', type=int, default=0) # 0 is expected ans, 1 is ans1, 2 is ans2
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=60)
    parser.add_argument('--kos', type=str, default='show') ###
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--p_value', type=float, default=0.49)
    parser.add_argument('--add', type=str, default='')
    
    global args
    args = parser.parse_args()
    # t = strftime('%Y%m%d-%H%M')

    args.T = strftime('%Y%m%d-%H%M')
        
    print(args)

    know_bias_analysis_temp_all_new(args.model_name, args.condition, args.num_probs)


if __name__ == '__main__':
    main()