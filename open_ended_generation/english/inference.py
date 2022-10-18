# coding=utf-8
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

def inference_one_instance(args, data, index, eos_token_id, model, cuda_available, device):
    decoding_method = args.decoding_method
    assert decoding_method in ['greedy', 'beam', 'topk', 'nucleus', 'contrastive']

    input_ids = data.prefix_token_id_list[index]
    input_ids = torch.LongTensor(input_ids).view(1,-1)
    _, prefix_len = input_ids.size()
    if cuda_available:
        input_ids = input_ids.cuda(device)

    decoding_len = args.decoding_len
    all_output_text_list = []
    with torch.no_grad():
        if decoding_method == 'greedy':
            number_of_instance_to_generate_per_method = 1
            output = model.greedy_search(input_ids=input_ids, decoding_len=decoding_len,
                end_of_sequence_token_id = eos_token_id, early_stop = True)
            output_text = model.tokenizer.decode(output[prefix_len:])
            all_output_text_list = [output_text]
        elif decoding_method == 'beam':
            number_of_instance_to_generate_per_method = 1
            output = model.beam_search(input_ids=input_ids, beam_width=4, decoding_len=decoding_len, 
                end_of_sequence_token_id = eos_token_id, early_stop = True)
            output_text = model.tokenizer.decode(output[prefix_len:])
            all_output_text_list = [output_text]
        elif decoding_method == 'contrastive':
            number_of_instance_to_generate_per_method = 1
            k, alpha = 5, 0.6
            output = model.fast_contrastive_search(input_ids=input_ids, beam_width=k, alpha=alpha, 
                        decoding_len=decoding_len, end_of_sequence_token_id = eos_token_id, early_stop = True)
            output_text = model.tokenizer.decode(output[prefix_len:])
            all_output_text_list = [output_text]
        else:
            number_of_instance_to_generate_per_method = args.number_of_instance_to_generate_per_method
            for instance_idx in range(number_of_instance_to_generate_per_method):
                if decoding_method == 'topk':
                    output = model.topk_sampling(input_ids=input_ids, topk=50, decoding_len=decoding_len, 
                        end_of_sequence_token_id = eos_token_id, early_stop = True)
                    output_text = model.tokenizer.decode(output[prefix_len:])
                    all_output_text_list.append(output_text)
                elif decoding_method == 'nucleus':
                    output = model.nucleus_sampling(input_ids=input_ids, nucleus_p=0.95, decoding_len=decoding_len, 
                        end_of_sequence_token_id = eos_token_id, early_stop = True)
                    output_text = model.tokenizer.decode(output[prefix_len:])
                    all_output_text_list.append(output_text)
                else:
                    raise Exception('Wrong decoding mode!!!')

    res_dict = {}
    res_dict['prefix_text'] = data.prefix_text_list[index]
    res_dict['reference_text'] = data.reference_text_list[index]

    generated_dict = {}
    for one_idx in range(number_of_instance_to_generate_per_method):
        generated_dict[one_idx] = all_output_text_list[one_idx]
    res_dict['generated_result'] = generated_dict
    return res_dict

def parse_config():
    parser = argparse.ArgumentParser()
    # model and data configuration
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--data_path", type=str)
    # decoding configuration
    parser.add_argument("--decoding_method", type=str)
    parser.add_argument("--prefix_len", type=int)
    parser.add_argument("--decoding_len", type=int)
    parser.add_argument("--number_of_instance_to_generate_per_method", type=int, default=1)
    # save configuration
    parser.add_argument("--save_path_prefix", type=str)
    return parser.parse_args()

if __name__ == '__main__':
    if torch.cuda.is_available():
        print ('Cuda is available.')
    cuda_available = torch.cuda.is_available()
    args = parse_config()
    device = torch.device('cuda')

    save_path_prefix = args.save_path_prefix + '{}/{}/'.format(args.model_name, args.decoding_method)
    import os
    if os.path.exists(save_path_prefix):
        pass
    else: # recursively construct directory
        os.makedirs(save_path_prefix, exist_ok=True)
    save_name = '{}_result.json'.format(args.decoding_method)
    save_path = save_path_prefix + save_name
    print ('Result saving path is {}'.format(save_path))

    print ('Loading model...')
    from simctg.simctggpt import SimCTGGPT
    model_name = 'gpt2-large'
    model = SimCTGGPT(model_name)
    model.eval()
    tokenizer = model.tokenizer
    eos_token_id = tokenizer.eos_token_id
    if cuda_available:
        model = model.to(device)
    model.eval()
    print ('Model loaded.')

    print ('Loading data...')
    from dataclass import Data
    data = Data(tokenizer, args.prefix_len, args.decoding_len, args.data_path)
    print ('Data loaded.')

    print ('Performing inference...')
    data_num = min(1000, len(data.prefix_token_id_list))
    print (data_num)
    p = progressbar.ProgressBar(data_num)
    p.start()
    result_list = []
    with torch.no_grad():
        for index in range(data_num):
            p.update(index)
            one_res_dict = inference_one_instance(args, data, index, eos_token_id, 
                model, cuda_available, device)
            result_list.append(one_res_dict)
    p.finish()
    print ('Inference completed!')

    import json
    with open(save_path, 'w') as outfile:
        json.dump(result_list, outfile, indent=4)
