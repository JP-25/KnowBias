import os
import random
import csv
import tqdm
import argparse
import itertools
import logging
from time import strftime
import sys
from llm_utils import *
from prettytable import PrettyTable
import pandas as pd
from datasets import load_dataset
from kn_code.patch import *
from kn_code.plot import pre_post_edit_probs, plot
from kn_code import load_model


DATA_DIR = 'data'
PROMPT_DIR = 'prompt_instructions'
# T = strftime('%Y%m%d-%H%M')



def evaluate(all_cat, all_):
    data = args.dataset

    if "Llama" in args.model_name and 'kn' in args.neurons: # 4 bit load
        print('kn enhance llama loaded...')
        perspectiveModel = LLM_KN(args.model_name, temperature=args.temperature, load_in_8bit=args.eight_bit, verbose=args.verbose)
    elif "Qwen" in args.model_name and 'kn' in args.neurons: # 4 bit load
        print('kn enhance qwen loaded...')
        perspectiveModel = LLM_KN(args.model_name, temperature=args.temperature, load_in_8bit=args.eight_bit, verbose=args.verbose)
    elif "gemma" in args.model_name and 'kn' in args.neurons: # 4 bit load
        print('kn enhance gemma loaded...')
        perspectiveModel = LLM_KN(args.model_name, temperature=args.temperature, load_in_8bit=args.eight_bit, verbose=args.verbose)
    elif "Llama" in args.model_name or 'Qwen' in args.model_name or 'gemma' in args.model_name:
        perspectiveModel = LLM_KN(args.model_name, temperature=args.temperature, load_in_8bit=args.eight_bit, verbose=args.verbose) # this one to load model

    table = PrettyTable()
    
    print("\n------------------------")
    print("    EVALUATING WITH      ")
    print("------------------------")

    print(f"MODEL: {args.model_name}")
    print(f"CAT: {str(all_cat)}")
    print("------------------------\n")

        
    if args.model_name == 'meta-llama/Llama-3.2-3B-Instruct':
        print('llama-3.2-3B loaded!')
        
        para = args.para
        neurons = [(0, 279), (0, 7345), (1, 1419), (1, 6495), (2, 2691), (2, 6541), (3, 2204), (3, 2897), (3, 3322), (4, 4), (4, 3176), (4, 4199), (4, 4968), (5, 3185), (5, 3982), (5, 7921), (5, 8072), (6, 1400), (6, 2517), (6, 2644), (6, 3357), (6, 3603), (6, 3766), (6, 6863), (6, 6866), (6, 6952), (6, 7105), (7, 267), (7, 1363), (7, 3521), (7, 3829), (7, 4349), (7, 6103), (7, 8110), (7, 8157), (8, 355), (8, 365), (8, 493), (8, 672), (8, 798), (8, 998), (8, 1147), (8, 1304), (8, 1388), (8, 1474), (8, 1547), (8, 1683), (8, 2303), (8, 3118), (8, 3400), (8, 3447), (8, 3642), (8, 3766), (8, 4267), (8, 4574), (8, 4693), (8, 4872), (8, 5298), (8, 5432), (8, 5624), (8, 5769), (8, 6421), (8, 6490), (8, 6850), (8, 6944), (8, 7068), (8, 7271), (8, 7501), (8, 8066), (9, 1365), (9, 1991), (9, 2384), (9, 5303), (9, 5334), (9, 6057), (9, 6067), (10, 208), (10, 411), (10, 782), (10, 1094), (10, 1303), (10, 1614), (10, 1696), (10, 1760), (10, 1869), (10, 2010), (10, 2084), (10, 2320), (10, 2609), (10, 3017), (10, 3182), (10, 3306), (10, 3574), (10, 3700), (10, 4025), (10, 4531), (10, 4710), (10, 5562), (10, 5985), (10, 6021), (10, 6048), (10, 6101), (10, 6199), (10, 6333), (10, 6464), (10, 6485), (10, 7336), (10, 7486), (10, 7531), (10, 7830), (10, 8020), (10, 8070), (11, 49), (11, 224), (11, 719), (11, 1037), (11, 1397), (11, 1413), (11, 1534), (11, 1665), (11, 1719), (11, 1825), (11, 2087), (11, 2298), (11, 2427), (11, 2691), (11, 2802), (11, 2821), (11, 2960), (11, 2982), (11, 3128), (11, 3137), (11, 3524), (11, 3552), (11, 3625), (11, 3725), (11, 3798), (11, 3889), (11, 4065), (11, 4385), (11, 4868), (11, 5166), (11, 5293), (11, 5404), (11, 5504), (11, 5588), (11, 5692), (11, 5878), (11, 5903), (11, 6125), (11, 6463), (11, 6829), (11, 7041), (11, 7291), (11, 7320), (11, 7325), (11, 7514), (11, 7649), (11, 7932), (12, 93), (12, 359), (12, 603), (12, 664), (12, 885), (12, 899), (12, 907), (12, 1148), (12, 1159), (12, 1215), (12, 1345), (12, 1819), (12, 2023), (12, 2167), (12, 2416), (12, 2711), (12, 2816), (12, 3107), (12, 3127), (12, 3322), (12, 3642), (12, 3842), (12, 3996), (12, 4065), (12, 4191), (12, 4720), (12, 4888), (12, 4895), (12, 5112), (12, 6180), (12, 6283), (12, 6362), (12, 6386), (12, 6410), (12, 6510), (12, 6599), (12, 6833), (12, 7581), (12, 7796), (12, 7810), (12, 7816), (12, 7943), (12, 7967), (12, 8119), (13, 67), (13, 509), (13, 858), (13, 1358), (13, 3076), (13, 3928), (13, 4327), (13, 4963), (13, 5235), (13, 6107), (13, 6118), (14, 3877), (14, 6428), (14, 6616), (14, 7690), (14, 8108), (15, 138), (15, 142), (15, 360), (15, 397), (15, 637), (15, 743), (15, 803), (15, 868), (15, 888), (15, 1080), (15, 1156), (15, 1198), (15, 1271), (15, 1391), (15, 1546), (15, 1591), (15, 1676), (15, 1739), (15, 2254), (15, 2347), (15, 2674), (15, 2685), (15, 2856), (15, 2931), (15, 3280), (15, 3296), (15, 3323), (15, 3626), (15, 4318), (15, 4352), (15, 4602), (15, 4689), (15, 4824), (15, 4912), (15, 5197), (15, 5245), (15, 5747), (15, 5759), (15, 5770), (15, 5837), (15, 5884), (15, 6075), (15, 6088), (15, 6221), (15, 6275), (15, 6502), (15, 6622), (15, 6655), (15, 6711), (15, 7067), (15, 7389), (15, 7392), (15, 7498), (15, 7698), (15, 7741), (15, 7945), (16, 93), (16, 734), (16, 1193), (16, 2499), (16, 3827), (16, 4272), (16, 4858), (16, 5477), (16, 5690), (16, 6309), (16, 6499), (16, 7015), (16, 7436), (17, 384), (17, 637), (17, 941), (17, 1226), (17, 2012), (17, 2018), (17, 2257), (17, 3069), (17, 4065), (17, 4242), (17, 4707), (17, 5575), (17, 6723), (17, 8006), (18, 839), (18, 858), (18, 2266), (18, 2297), (18, 2840), (18, 3655), (18, 4917), (19, 734), (19, 965), (19, 1504), (19, 6955), (19, 7405), (20, 5927), (20, 5939), (20, 6131), (21, 12), (21, 77), (21, 189), (21, 1081), (21, 1627), (21, 2277), (21, 4448), (21, 8190), (22, 549), (22, 718), (22, 2058), (22, 2656), (23, 7), (23, 966), (23, 1139), (23, 1711), (23, 3277), (23, 3325), (23, 4422), (23, 4561), (23, 7440), (24, 871), (24, 2363), (24, 2754), (24, 3514), (24, 3859), (24, 6206), (24, 6216), (24, 7082), (24, 7359), (24, 7486), (24, 7519), (24, 7931), (25, 3910), (25, 4349), (25, 5254), (26, 198), (26, 245), (26, 593), (26, 997), (26, 1281), (26, 1607), (26, 2500), (26, 3077), (26, 3297), (26, 3812), (26, 3830), (26, 4158), (26, 5865), (26, 6100), (26, 6438), (26, 6441), (26, 6466), (26, 6534), (26, 6594), (26, 6815), (26, 7963), (27, 883), (27, 1055), (27, 2824), (27, 4912), (27, 5230), (27, 5404), (27, 5865), (27, 7034), (27, 7254), (27, 7597)]


    if args.neurons == 'kn_enhance':
        modified_layers = perspectiveModel.kn.enhance(neurons, val=args.enhance_val)
    elif args.neurons == 'kn_suppress':
        modified_layers = perspectiveModel.kn.erase_0(neurons)

    if data == 'bbq':
        print("\n------------------------")
        print(f"CONDITION: {args.condition}")
        print("------------------------")

        table.field_names = ["question", "gold", "predictions", "True", "Wrong", "unknown", 'bias cat', 'bias target', 'id', 'q_index', 'q_polarity']

        predicted_answers = []
        neurons = []

        for cat in all_cat:
            bbq_subset = load_dataset("heegyu/bbq", cat)
            all_rows = bbq_subset['test']

            if all_:
                args.num_probs = len(all_rows)
            
            args.num_probs = min(args.num_probs, len(all_rows))
            
            unknown_label = None
            for index, row in enumerate(tqdm.tqdm(all_rows)):
                id = row['example_id']
                q_p = row['question_polarity']
                q_index = row['question_index']
                context_con = row['context_condition']
                bias_target = row['additional_metadata']['stereotyped_groups']

                if context_con == 'ambig':
                    unknown_label = row['label']

                if context_con != args.condition: # answer label should be anti-stereotype
                    continue

                ans0 = row['ans0']
                ans1 = row['ans1']
                ans2 = row['ans2']
                label = row['label']

                answers_ori = [ans0, ans1, ans2]
                true_answer = answers_ori[label]
                if context_con == 'ambig':
                    wrong_labels = list(set({0, 1, 2}) - set({label}))
                    # this case, unknown served as one of the wrong answers
                    unknown_answer = answers_ori[wrong_labels[0]]
                    wrong_answer = answers_ori[wrong_labels[1]]
                else:
                    unknown_answer = answers_ori[unknown_label] # include or not
                    wrong_label = list(set({0, 1, 2}) - set({label, unknown_label}))[0]
                    wrong_answer = answers_ori[wrong_label]

                # answers = [true_answer, wrong_answer]
                answers = [true_answer, wrong_answer, unknown_answer]

                random.shuffle(answers)
                context = row['context']

                bias_type = row['additional_metadata']['stereotyped_groups']
                q = row['question']

                question = f"{q}\nChoose from ({answers[0]};{answers[1]};{answers[2]})"

                with(open(f'{PROMPT_DIR}/evaluate_s.txt', 'r')) as f:
                    instruction = f.read()
                
                # Base prompt. The instruction corresponds to the above instruction prompts for diff. methods.
                prompt = f"""{instruction}
Context: {context}
Question: {question}"""
                
                predicted_answer = perspectiveModel.getOutput(prompt)

                predicted_answers.append(predicted_answer)

                table.add_row([prompt, true_answer, predicted_answer, true_answer, wrong_answer, unknown_answer, cat, bias_target, id, q_index, q_p])
    
    elif data == 'crows':

        if args.type == 'gender':
            b_t = 'gender'
        elif args.type == 'race':
            b_t = 'race-color'
        elif args.type == 'religion':
            b_t = 'religion'

        cat = 'crows_' + b_t

        predicted_answers = []
        
        table.field_names = ["question", "gold", "non-gold", "True", "Wrong", "unknown", 'bias cat', 'bias target', 'q_index', 'q_polarity', 'bias_res']
        csv_name = os.path.join(DATA_DIR, 'crows_pairs_anonymized.csv')
        with open(csv_name, "r") as f:
            reader = csv.reader(f, delimiter=',')
            all_rows = list(reader)

        if all_:
            args.num_probs = len(all_rows)
        
        args.num_probs = min(args.num_probs, len(all_rows))

        for index, row in enumerate(tqdm.tqdm(all_rows[1:args.num_probs])):
            id = row[0]
            sent_more = row[1]
            sent_less = row[2]
            bias_cat = row[4]
            bias_target = row[5]

            if bias_cat != b_t: #
                continue

            true_answer = sent_less
            wrong_answer = sent_more

            answers = [true_answer, wrong_answer]

            with(open(f'{PROMPT_DIR}/evaluate_probs.txt', 'r')) as f:
                instruction = f.read()

            question = f"Choose one of the following that you think is the most appropriate:\n{answers[0]}\n{answers[1]}"

            prompt = f"""{instruction}

{question}"""
            
            predicted_answer_0 = perspectiveModel.getProb(prompt, answers[0])
            predicted_answer_1 = perspectiveModel.getProb(prompt, answers[1])

            table.add_row([prompt, true_answer, wrong_answer, predicted_answer_0, predicted_answer_1, 'na', bias_cat, bias_target, index, 'na', 'na'])
    
    elif data == 'stereo_intra' or data == 'stereo_inter':

        if args.type == 'gender':
            b_t = 'gender'
        elif args.type == 'race':
            b_t = 'race'
        elif args.type == 'religion':
            b_t = 'religion'

        cat = 'stereoset_' + ('intra' if data == 'stereo_intra' else 'inter') + '_' + b_t

        if data == 'stereo_intra':
            stereo_sub = load_dataset("McGill-NLP/stereoset", "intrasentence")
        else:
            stereo_sub = load_dataset("McGill-NLP/stereoset", "intersentence")
        all_rows = stereo_sub['validation']
        if all_:
            args.num_probs = len(all_rows)

        table.field_names = ["question", "gold", "non-gold", "True", "Wrong", "unknown", 'bias cat', 'bias target', 'q_index', 'q_polarity', 'bias_res']

        for index, row in enumerate(tqdm.tqdm(all_rows)):
            id = row['id']
            bias_type = row['bias_type']
            target = row['target']
            gold_label = row['sentences']['gold_label']
            stereotype_index = gold_label.index(1) # stereotype sentence idx
            anti_stereotype_index = gold_label.index(0) # anti stereotype sentence idx
            no_related_index = gold_label.index(2)

            if bias_type != b_t:
                continue

            sent_more = row['sentences']['sentence'][stereotype_index]
            sent_less = row['sentences']['sentence'][anti_stereotype_index]
            no_related = row['sentences']['sentence'][no_related_index]

            context = row['context']

            true_answer = sent_less
            wrong_answer = sent_more
            unknown_answer = no_related
            answers = [true_answer, wrong_answer, unknown_answer]

            question = f"Choose one of the following:\n{answers[0]}\n{answers[1]}\n{answers[2]}"

            with(open(f'{PROMPT_DIR}/evaluate_probs.txt', 'r')) as f:
                instruction = f.read()
            
            # Base prompt. The instruction corresponds to the above instruction prompts for diff. methods.
            prompt = f"""{instruction}

Context: {context}
{question}"""
            
            predicted_answer_0 = perspectiveModel.getProb(prompt, answers[0])
            predicted_answer_1 = perspectiveModel.getProb(prompt, answers[1])
            predicted_answer_2 = perspectiveModel.getProb(prompt, answers[2])

            table.add_row([prompt, 'na', 'na', predicted_answer_0, predicted_answer_1, predicted_answer_2, bias_type, target, index, 'na', 'na'])

    if 'kn' in args.neurons:
        print('Unpatching model...')
        unpatch_ff_layers(
            model=perspectiveModel.kn.model,
            layer_indices=modified_layers,
            transformer_layers_attr=perspectiveModel.kn.transformer_layers_attr,
            ff_attrs=perspectiveModel.kn.input_ff_attr,
        )


    if 'kn' in args.neurons:

        if args.dataset == 'crows':
            cat = 'crows_' + args.type + '_' + args.add_info + '_' + str(para) + '_' + str(args.enhance_val) # think, is, f

        if args.dataset == 'bbq':
            cat = 'bbq_' + args.type + '_' + args.add_info + '_' + str(para) + '_' + str(args.enhance_val) # think, is, f

        if args.dataset == 'stereo_intra':
            cat = 'stereo_intra_' + args.type + '_' + args.add_info + '_' + str(para) + '_' + str(args.enhance_val) # think, is, f
        
        if args.dataset == 'stereo_inter':
            cat = 'stereo_inter_' + args.type + '_' + args.add_info + '_' + str(para) + '_' + str(args.enhance_val) # think, is, f

    
    concept_dir = os.path.join('final_res', cat)
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
    
    save_concepts = pd.DataFrame(table._rows, columns=table.field_names)

    # t = strftime('%Y%m%d-%H%M')
    temp = str(args.temperature)

    save_concepts_dir = os.path.join(concept_dir, args.T + '_' + cat + '_' + model_name + '_' + args.dataset + '_' + args.condition + '_results.csv')

    save_concepts.to_csv(save_concepts_dir, index = False, header=True)
    
    # Print results
    print("\n------------------------")
    print("         RESULTS        ")
    print("------------------------")

    print(f"MODEL: {args.model_name}")
    print(f"CONDITION: {args.condition}")

    print("------------------------\n")
    

def main():
    # python evaluate.py --model_name=meta-llama/Meta-Llama-3.1-8B-Instruct --dataset=bbq --num_probs=10
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', type=str, default='ambig') # disambig, ambig
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--num_probs', '-n', type=int, default=200)
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--gpu', type=int, default=0) # which gpu to load on
    parser.add_argument('--eight_bit', action='store_true') # load model in 8-bit?
    parser.add_argument('--dataset', type=str, default='crows') # choose from crows, stereo_intra, stereo_inter, bbq_<category>
    parser.add_argument('--T', type=str, default="")
    parser.add_argument('--neurons', type=str, default='na') # kn, kn_enhance, kn_suppress
    parser.add_argument('--op', type=int, default=-1) # 1, ans a, 2, ans b, 0 all
    parser.add_argument('--part', type=int, default=-1) # split running for sofa
    parser.add_argument('--type', type=str, default='') # gender, race, religion
    parser.add_argument('--enhance_val', type=float, default=1) # kn_enhance_val
    parser.add_argument('--add_info', type=str, default='') # which layer to enhance for kn_enhance
    parser.add_argument('--para', type=float, default=0.1) # bias, common_sense
    # parser.add_argument('--bias_cat', type=str, default='Age') #
    
    global args
    args = parser.parse_args()

    # args.T = strftime('%Y%m%d-%H%M')
    args.T = strftime('%Y%m%d-%H%M%S') # to seconds
    
    print(args)
    if args.type == 'gender':
        all_cat = ['Gender_identity']
    elif args.type == 'race':
        all_cat = all_cat = ['Race_ethnicity']
    elif args.type == 'religion':
        all_cat = all_cat = ['Religion']
    else:
        all_cat = ['Gender_identity', 'Race_ethnicity', 'Religion']
    
    evaluate(all_cat, args.all)

if __name__ == '__main__':
    main()