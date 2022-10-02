# coding=utf-8
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import argparse, os
import random
import numpy as np
import time
import logging
import progressbar

import logging
logging.getLogger('transformers.generation_utils').disabled = True

def parse_output(output, prefix_len, tokenizer):
    # output: a list of token ids
    output_text = tokenizer.decode(output[prefix_len:], 
        truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n", "\n\ndef"])
    return output_text

def generate_one_instance(args, test_item, model, tokenizer, cuda_available, device):
    evaluation_method = args.evaluation_method
    assert evaluation_method in ['greedy', 'beam', 'nucleus', 'contrastive']

    input_ids = test_item['prompt_ids']
    _, prefix_len = input_ids.size()
    if cuda_available:
        input_ids = input_ids.cuda(device)

    decoding_len = args.decoding_len
    with torch.no_grad():
        if evaluation_method == 'greedy':
            output = model.model.generate(input_ids=input_ids, max_length=prefix_len+decoding_len, 
                early_stopping=False, eos_token_id=-1)[0]
            output = output.detach().cpu()
        elif evaluation_method == 'beam':
            output = model.model.generate(input_ids=input_ids, max_length=prefix_len+decoding_len, 
                early_stopping=False, eos_token_id=-1,
                num_beams=5)[0]
            output = output.detach().cpu()
        elif evaluation_method == 'nucleus':
            output = model.model.generate(input_ids=input_ids, max_length=prefix_len+decoding_len, 
                early_stopping=False, eos_token_id=-1,
                do_sample=True,
                top_p=0.95,
                top_k=0)[0]
            output = output.detach().cpu()
        elif evaluation_method == 'contrastive':
            k, alpha = 3, 0.4
            output = model.fast_contrastive_search(input_ids=input_ids, beam_width=k, 
                                           alpha=alpha, decoding_len=decoding_len) 
            #output = output.detach().cpu()
        else:
            raise Exception('Wrong evaluation mode!!!')

    output_text = parse_output(output, prefix_len, tokenizer)
    result_dict = {
        'task_id':test_item['task_id'],
        'completion':output_text
    }
    return result_dict

def save_result(result_dict_list, save_path_prefix, save_name):
    if os.path.exists(save_path_prefix):
        pass
    else: # recursively construct directory
        os.makedirs(save_path_prefix, exist_ok=True)

    save_path = save_path_prefix + save_name
    import json
    with open(save_path, 'w') as outfile:
        for entry in result_dict_list:
            json.dump(entry, outfile)
            outfile.write('\n')
    return save_path

import argparse
def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--test_path", type=str)
    parser.add_argument("--evaluation_method", type=str)
    parser.add_argument("--decoding_len", type=int)
    parser.add_argument("--run_num", type=int)
    parser.add_argument("--save_path_prefix")
    return parser.parse_args()

if __name__ == '__main__':
    if torch.cuda.is_available():
        print ('Cuda is available.')
    cuda_available = torch.cuda.is_available()
    args = parse_config()
    device = torch.device('cuda')

    print ('Loading model...')
    from models.simctgcodegen import SimCTGCodeGen
    model = SimCTGCodeGen(args.model_name)
    codegen_name = args.model_name[11:]
    print ('CodeGen model name is {}'.format(codegen_name))
    if cuda_available:
        model = model.to(device)
    model.eval()
    tokenizer = model.tokenizer
    print ('Model loaded.')

    print ('Loading data...')
    from dataclass import Data
    data = Data(args.test_path, tokenizer)
    print ('Data loaded.')

    # prepare save directory
    import os
    save_path_prefix = args.save_path_prefix + r'/{}/{}/'.format(codegen_name, 
        args.evaluation_method)
    if os.path.exists(save_path_prefix):
        pass
    else: # recursively construct directory
        os.makedirs(save_path_prefix, exist_ok=True)

    save_name = r'{}-{}-run-{}.jsonl'.format(codegen_name, args.evaluation_method, args.run_num)
    save_path = save_path_prefix + save_name
    print ('Save path is {}'.format(save_path))

    print ('Performing inference...')
    data_num = len(data.data_list)
    p = progressbar.ProgressBar(data_num)
    p.start()
    result_list = []
    with torch.no_grad():
        for index in range(data_num):
            p.update(index)

            test_item = data.data_list[index]
            one_result_dict = generate_one_instance(args, test_item, model, 
                tokenizer, cuda_available, device)
            result_list.append(one_result_dict)
    p.finish()
    print ('Inference completed!')

    print ('Saving predictions...')
    save_result(result_list, save_path_prefix, save_name)
    print ('Prediction saved!')

    print ('Perform evaluation...')
    problem_file_path = args.test_path
    prediction_file_path = save_path
    from evaluation import evaluate_pass_score
    pass_score = evaluate_pass_score(problem_file_path, prediction_file_path)
    print ('Evaluation completed!')

    evaluation_result_dict = {'pass@1':pass_score}
    print ('Evaluation result is ')
    print (evaluation_result_dict)

    evaluation_save_name = r'{}-{}-run-{}-evaluation-result.json'.format(codegen_name, 
        args.evaluation_method, args.run_num)
    evaluation_save_path = save_path_prefix + evaluation_save_name

    with open(evaluation_save_path, 'w') as outfile:
        json.dump([evaluation_result_dict], outfile, indent=4)
